"""
Offline A/B comparison of two registered model versions.

Setup
-----
Two completed training runs are two model *versions*; their run ids are the
"version ids". The A/B flow (ab_flow.py) takes those two run ids as
configuration and forks into two branches, one per version. Each branch calls
``score`` for its version, then a join step calls ``compare`` on both outputs.

Splitting the traffic (random yet reproducible)
-----------------------------------------------
Each row carries a stable identifier, ``Animal ID``. We assign a row to a
variant by hashing ``"<test_id>:<animal_id>"`` with MD5 and taking the result
mod 2 (bucket 0 -> variant A, bucket 1 -> variant B). This is:

  * **deterministic / reproducible** -- the same id always lands in the same
    bucket for a given test, no RNG seed or stored assignment table needed;
  * **roughly even** -- MD5 is uniform, so ~50/50 over many ids;
  * **independent of row order or arrival time** -- it is a pure function of the
    id, so re-running scores the exact same partition.

Why salt with the test id?
--------------------------
Hashing ``"<test_id>:<animal_id>"`` rather than the bare id means each test gets
its *own* pseudo-random partition. Without the salt, the same animals would
always sit in bucket 0 across every test, so a unit that did well under one
test's variant A would keep landing in variant A everywhere -- assignments would
be correlated across tests. The salt decorrelates them while staying
reproducible within a test.

Multiple concurrent or subsequent tests
---------------------------------------
Give every experiment a distinct ``test_id`` and carry it on both the split (as
the hash salt, above) and the results (every report records its ``test_id``).
That single key does double duty: it isolates each test's 50/50 partition from
the others, and it namespaces the stored outcomes so concurrent tests never mix.

Deciding a winner
-----------------
The split simulates online traffic, where each animal only ever sees one
version. Each half is about 5,800 rows, so the two halves differ by chance by
a few points of macro F1 even when the models are identical. The first version
of this module declared the version with the higher number the winner; two
models that disagreed on 96 of 11,812 rows were reported as "A wins by
0.037 macro F1", which was noise.

Offline we can do better, because both versions can score every row. So each
branch scores the whole segment, and ``compare`` reports two things:

  * **split**: A on bucket 0 against B on bucket 1, the online view, with a
    bootstrap 95% interval that resamples each half independently;
  * **paired**: both versions on every row, with a bootstrap 95% interval that
    resamples rows and keeps each row's two predictions together. Pairing
    removes the row to row noise, so this interval is far narrower.

A winner is only declared when the paired interval for the macro F1 difference
excludes zero; otherwise the result is "no significant difference".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import data as data_mod
from . import registry as reg

_VARIANT_BUCKET = {"A": 0, "B": 1}
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


@dataclass
class VariantReport:
    test_id: str
    variant: str
    run_id: str
    segment_year: int
    n_rows: int
    accuracy: float
    macro_f1: float


def assign_bucket(animal_id: str, test_id: str) -> int:
    """Map an id to bucket 0/1 deterministically, salted by the test id."""
    digest = hashlib.md5(f"{test_id}:{animal_id}".encode()).hexdigest()
    return int(digest, 16) % 2


def score_variant(run_id: str, variant: str, test_id: str,
                  segment_year: int) -> tuple[VariantReport, pd.DataFrame]:
    """Score one version on the whole segment.

    Returns the report for the version's own half (the online view) and the
    per row predictions for every row, which ``compare`` needs for the paired
    comparison.
    """
    variant = variant.upper()
    if variant not in _VARIANT_BUCKET:
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    bucket = _VARIANT_BUCKET[variant]

    df = data_mod.load_year(segment_year)
    X, y, ids = data_mod.split_xy_id(df)
    buckets = ids.map(lambda i: assign_bucket(i, test_id)).to_numpy()
    mask = buckets == bucket
    if not mask.any():
        raise RuntimeError(
            f"variant {variant} got 0 rows for test {test_id!r}; check the split"
        )

    preds = reg.load(run_id).predict(X)
    rows = pd.DataFrame({"animal_id": ids.to_numpy(), "bucket": buckets,
                         "y_true": y.to_numpy(), "y_pred": preds})
    report = VariantReport(
        test_id=test_id,
        variant=variant,
        run_id=run_id,
        segment_year=segment_year,
        n_rows=int(mask.sum()),
        accuracy=float(accuracy_score(y[mask], preds[mask])),
        macro_f1=float(f1_score(y[mask], preds[mask], average="macro", zero_division=0)),
    )
    return report, rows


# --- comparison ---------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro F1 over integer labels, matching sklearn's default (labels present
    in either y_true or y_pred, zero_division=0). Vectorised for the bootstrap."""
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes)
    denom = np.bincount(y_true, minlength=n_classes) + np.bincount(y_pred, minlength=n_classes)
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def _interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def compare(rows_a: pd.DataFrame, rows_b: pd.DataFrame, test_id: str,
            n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Compare two versions scored on the same segment (see module docstring)."""
    if not (rows_a["animal_id"].to_numpy() == rows_b["animal_id"].to_numpy()).all():
        raise RuntimeError("the two variants scored different rows; cannot pair them")
    labels = sorted(set(rows_a["y_true"]) | set(rows_a["y_pred"]) | set(rows_b["y_pred"]))
    code = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    y = rows_a["y_true"].map(code).to_numpy()
    pa = rows_a["y_pred"].map(code).to_numpy()
    pb = rows_b["y_pred"].map(code).to_numpy()
    rng = np.random.default_rng(seed)

    # Split view: A on bucket 0, B on bucket 1, halves resampled independently.
    ia = np.flatnonzero(rows_a["bucket"].to_numpy() == 0)
    ib = np.flatnonzero(rows_a["bucket"].to_numpy() == 1)
    split_f1 = _macro_f1(y[ib], pb[ib], k) - _macro_f1(y[ia], pa[ia], k)
    split_acc = float((y[ib] == pb[ib]).mean() - (y[ia] == pa[ia]).mean())
    boot_f1, boot_acc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(ia, size=ia.size)
        sb = rng.choice(ib, size=ib.size)
        boot_f1[i] = _macro_f1(y[sb], pb[sb], k) - _macro_f1(y[sa], pa[sa], k)
        boot_acc[i] = (y[sb] == pb[sb]).mean() - (y[sa] == pa[sa]).mean()
    split = {
        "n_rows_a": int(ia.size), "n_rows_b": int(ib.size),
        "macro_f1_b_minus_a": split_f1, "macro_f1_b_minus_a_ci95": _interval(boot_f1),
        "accuracy_b_minus_a": split_acc, "accuracy_b_minus_a_ci95": _interval(boot_acc),
    }

    # Paired view: both versions on every row, rows resampled together.
    n = y.size
    paired_f1 = _macro_f1(y, pb, k) - _macro_f1(y, pa, k)
    paired_acc = float((y == pb).mean() - (y == pa).mean())
    for i in range(n_boot):
        s = rng.integers(0, n, size=n)
        boot_f1[i] = _macro_f1(y[s], pb[s], k) - _macro_f1(y[s], pa[s], k)
        boot_acc[i] = (y[s] == pb[s]).mean() - (y[s] == pa[s]).mean()
    paired_ci = _interval(boot_f1)
    paired = {
        "n_rows": int(n),
        "rows_where_versions_disagree": int((pa != pb).sum()),
        "macro_f1_a": _macro_f1(y, pa, k), "macro_f1_b": _macro_f1(y, pb, k),
        "accuracy_a": float((y == pa).mean()), "accuracy_b": float((y == pb).mean()),
        "macro_f1_b_minus_a": paired_f1, "macro_f1_b_minus_a_ci95": paired_ci,
        "accuracy_b_minus_a": paired_acc, "accuracy_b_minus_a_ci95": _interval(boot_acc),
    }

    if paired_ci[0] > 0:
        decision = "B"
    elif paired_ci[1] < 0:
        decision = "A"
    else:
        decision = "no significant difference"
    return {"test_id": test_id, "n_bootstrap": n_boot, "bootstrap_seed": seed,
            "split": split, "paired": paired, "decision_by_paired_macro_f1": decision}


# --- CLI ------------------------------------------------------------------------

def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="A/B scoring and comparison. Invoked by ab_flow.py inside "
                    "the ab container.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="Score one variant on the whole segment.")
    s.add_argument("--run-id", required=True, help="Model version id for this variant.")
    s.add_argument("--variant", required=True, choices=["A", "B", "a", "b"])
    s.add_argument("--test-id", required=True, help="Experiment id (also the split salt).")
    s.add_argument("--segment-year", type=int, default=2024)
    s.add_argument("--out-dir", required=True,
                   help="Writes variant_<a|b>.json and predictions_<a|b>.csv here.")
    c = sub.add_parser("compare", help="Compare the two scored variants.")
    c.add_argument("--dir", required=True, help="Directory holding both variants' outputs.")
    c.add_argument("--test-id", required=True)
    args = p.parse_args()

    if args.cmd == "score":
        report, rows = score_variant(args.run_id, args.variant, args.test_id, args.segment_year)
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        v = report.variant.lower()
        (out / f"variant_{v}.json").write_text(json.dumps(asdict(report), indent=2))
        rows.to_csv(out / f"predictions_{v}.csv", index=False)
        print(json.dumps(asdict(report), indent=2))
        return 0

    d = Path(args.dir)
    result = compare(pd.read_csv(d / "predictions_a.csv"),
                     pd.read_csv(d / "predictions_b.csv"), args.test_id)
    (d / "comparison_stats.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
