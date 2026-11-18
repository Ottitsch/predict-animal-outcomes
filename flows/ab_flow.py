"""
Metaflow flow: offline A/B comparison of two model versions.

Two registered model versions (two run ids) are compared head-to-head on a
shared unseen segment (default: the holdout year 2024). The flow forks into two
parallel branches -- one per version -- each of which scores the segment in an
isolated ``pao-ab`` container, then a join step compares them with bootstrap
confidence intervals (also in the ``pao-ab`` container) and only names a
winner when the difference is significant. See predict_animal_outcomes/ab_test.py.

Traffic split, salting, and running multiple tests are all explained in
predict_animal_outcomes/ab_test.py; in short, each row is assigned to a variant by hashing
``"<test_id>:<Animal ID>"`` mod 2, which is reproducible, ~50/50, and isolated
per test via the ``test_id`` salt.

Outputs are written under a fresh A/B run directory at
``runs/<ab-run-id>/ab/``: ``variant_a.json``, ``variant_b.json``, the per row
``predictions_a.csv`` and ``predictions_b.csv``, and the joined
``comparison.json``.

Run (containers):
    docker/build.sh
    docker run --rm \\
        -v /var/run/docker.sock:/var/run/docker.sock \\
        -v "$PWD":/work -w /work \\
        pao-host:dev flows/ab_flow.py run --run_id_a <id-a> --run_id_b <id-b> --test_id exp1

Run (host-native):
    USE_CONTAINERS=0 python flows/ab_flow.py run \\
        --run_id_a <id-a> --run_id_b <id-b> --test_id exp1
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# flows/ lives beside predict_animal_outcomes/; put the repo root on sys.path so `from predict_animal_outcomes ...`
# resolves regardless of launch dir (see flows/flow.py for the full rationale).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metaflow import FlowSpec, Parameter, step

from predict_animal_outcomes import containers


class ABTestFlow(FlowSpec):
    run_id_a = Parameter(
        "run_id_a", help="Run id of model version A.", required=True,
    )
    run_id_b = Parameter(
        "run_id_b", help="Run id of model version B.", required=True,
    )
    test_id = Parameter(
        "test_id",
        help="Experiment id. Salts the 50/50 split and tags the results so "
             "concurrent/subsequent tests stay isolated.",
        default="ab",
    )
    segment_year = Parameter(
        "segment_year",
        help="Unseen segment to split across the two variants (default: 2024).",
        default=2024,
        type=int,
    )

    @step
    def start(self):
        from predict_animal_outcomes import runs
        self.ab_run_id = runs.new_run_id()
        self.out_subpath = f"runs/{self.ab_run_id}/ab"
        (containers.ROOT / self.out_subpath).mkdir(parents=True, exist_ok=True)
        print("=== A/B Test Flow ===")
        print(f"  ab_run_id      = {self.ab_run_id}")
        print(f"  test_id        = {self.test_id}")
        print(f"  variant A      = {self.run_id_a}")
        print(f"  variant B      = {self.run_id_b}")
        print(f"  segment_year   = {self.segment_year}")
        print(f"  use_containers = {containers.USE_CONTAINERS}")
        self.next(self.branch_a, self.branch_b)

    @step
    def branch_a(self):
        self.variant_report = self._score("A", self.run_id_a)
        self.next(self.join)

    @step
    def branch_b(self):
        self.variant_report = self._score("B", self.run_id_b)
        self.next(self.join)

    def _score(self, variant: str, run_id: str) -> dict:
        out_dir = containers.ROOT / self.out_subpath
        stdout_log = out_dir / f"stdout_{variant.lower()}.log"
        cmd = [
            "python", "-m", "predict_animal_outcomes.ab_test", "score",
            "--run-id", run_id,
            "--variant", variant,
            "--test-id", self.test_id,
            "--segment-year", str(self.segment_year),
            "--out-dir", "/out" if containers.USE_CONTAINERS else str(out_dir),
        ]
        t0 = time.monotonic()
        if containers.USE_CONTAINERS:
            rc = containers.docker_run("ab", self.out_subpath, cmd, stdout_log)
        else:
            rc = containers.host_run(cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        if rc != 0:
            raise RuntimeError(f"variant {variant} scoring failed with exit={rc}")
        report = json.loads((out_dir / f"variant_{variant.lower()}.json").read_text())
        print(f"[branch_{variant.lower()}] acc={report['accuracy']:.4f} "
              f"macro_f1={report['macro_f1']:.4f} n={report['n_rows']} ({duration}s)")
        return report

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs, include=["ab_run_id", "out_subpath", "test_id"])
        reports = {inp.variant_report["variant"]: inp.variant_report for inp in inputs}
        a, b = reports["A"], reports["B"]
        out_dir = containers.ROOT / self.out_subpath
        # The bootstrap needs numpy, which the driver image doesn't carry, so the
        # comparison runs in the ab container like the scoring did.
        cmd = [
            "python", "-m", "predict_animal_outcomes.ab_test", "compare",
            "--dir", "/out" if containers.USE_CONTAINERS else str(out_dir),
            "--test-id", self.test_id,
        ]
        stdout_log = out_dir / "stdout_compare.log"
        if containers.USE_CONTAINERS:
            rc = containers.docker_run("ab", self.out_subpath, cmd, stdout_log)
        else:
            rc = containers.host_run(cmd, stdout_log)
        if rc != 0:
            raise RuntimeError(f"comparison failed with exit={rc}; see {stdout_log}")
        stats = json.loads((out_dir / "comparison_stats.json").read_text())
        (out_dir / "comparison_stats.json").unlink()
        comparison = {
            "test_id": self.test_id,
            "segment_year": a["segment_year"],
            "variant_a": a,
            "variant_b": b,
            **{k: v for k, v in stats.items() if k != "test_id"},
        }
        (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
        self.comparison = comparison
        self.next(self.end)

    @step
    def end(self):
        c = self.comparison
        p = c["paired"]
        lo, hi = p["macro_f1_b_minus_a_ci95"]
        print("=== A/B complete ===")
        print(f"  split  A: acc={c['variant_a']['accuracy']:.4f} f1={c['variant_a']['macro_f1']:.4f}"
              f"  (n={c['variant_a']['n_rows']})")
        print(f"  split  B: acc={c['variant_b']['accuracy']:.4f} f1={c['variant_b']['macro_f1']:.4f}"
              f"  (n={c['variant_b']['n_rows']})")
        print(f"  paired: macro F1 B-A = {p['macro_f1_b_minus_a']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  ({p['rows_where_versions_disagree']} of {p['n_rows']} rows differ)")
        print(f"  decision = {c['decision_by_paired_macro_f1']}")


if __name__ == "__main__":
    ABTestFlow()
