"""
Metaflow flow: offline A/B comparison of two model versions.

Two registered model versions (two run ids) are compared head-to-head on a
shared unseen segment (default: the holdout year 2024). The flow forks into two
parallel branches -- one per version -- each of which scores *its half* of the
segment in an isolated ``pao-ab`` container, then a join step compares the two
halves on accuracy and macro-F1.

Traffic split, salting, and running multiple tests are all explained in
src/ab_test.py; in short, each row is assigned to a variant by hashing
``"<test_id>:<Animal ID>"`` mod 2, which is reproducible, ~50/50, and isolated
per test via the ``test_id`` salt.

Outputs are written under a fresh A/B run directory at
``runs/<ab-run-id>/ab/``: ``variant_a.json``, ``variant_b.json`` and the joined
``comparison.json``.

Run (containers):
    docker/build.sh
    docker run --rm \\
        -v /var/run/docker.sock:/var/run/docker.sock \\
        -v "$PWD":/work -w /work \\
        pao-ab-host:dev run --run_id_a <id-a> --run_id_b <id-b> --test_id exp1

Run (host-native):
    USE_CONTAINERS=0 python ab_flow.py run \\
        --run_id_a <id-a> --run_id_b <id-b> --test_id exp1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from metaflow import FlowSpec, Parameter, step

from src import containers


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
        from src import runs
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
            "python", "-m", "src.ab_test",
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
        winner = (
            "tie" if a["macro_f1"] == b["macro_f1"]
            else "A" if a["macro_f1"] > b["macro_f1"] else "B"
        )
        comparison = {
            "test_id": self.test_id,
            "segment_year": a["segment_year"],
            "variant_a": a,
            "variant_b": b,
            "deltas": {
                "accuracy_b_minus_a": b["accuracy"] - a["accuracy"],
                "macro_f1_b_minus_a": b["macro_f1"] - a["macro_f1"],
            },
            "winner_by_macro_f1": winner,
        }
        out_dir = containers.ROOT / self.out_subpath
        (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
        self.comparison = comparison
        self.next(self.end)

    @step
    def end(self):
        c = self.comparison
        print("=== A/B complete ===")
        print(f"  A: acc={c['variant_a']['accuracy']:.4f} f1={c['variant_a']['macro_f1']:.4f}")
        print(f"  B: acc={c['variant_b']['accuracy']:.4f} f1={c['variant_b']['macro_f1']:.4f}")
        print(f"  winner_by_macro_f1 = {c['winner_by_macro_f1']}")


if __name__ == "__main__":
    ABTestFlow()
