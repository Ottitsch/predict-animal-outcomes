"""
Offline A/B scoring for two registered model versions.

Setup
-----
Two completed training runs are two model *versions*; their run ids are the
"version ids". The A/B flow (ab_flow.py) takes those two run ids as
configuration and forks into two branches, one per version. Each branch calls
this module to score *its half* of a shared unseen segment, and a join step
compares the two halves.

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
To read back "how did test X do", filter results by ``test_id``; a unit's
membership in any other test is irrelevant because each test salts its own
buckets. Subsequent tests are the same story over time -- a fresh ``test_id``
per launch keeps each test's partition and metrics cleanly separable.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from sklearn.metrics import accuracy_score, f1_score

from . import data as data_mod
from . import registry as reg

_VARIANT_BUCKET = {"A": 0, "B": 1}


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
                  segment_year: int) -> VariantReport:
    variant = variant.upper()
    if variant not in _VARIANT_BUCKET:
        raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
    bucket = _VARIANT_BUCKET[variant]

    df = data_mod.load_year(segment_year)
    X, y, ids = data_mod.split_xy_id(df)
    mask = ids.map(lambda i: assign_bucket(i, test_id) == bucket).to_numpy()
    X_v, y_v = X[mask], y[mask]
    if len(X_v) == 0:
        raise RuntimeError(
            f"variant {variant} got 0 rows for test {test_id!r}; check the split"
        )

    model = reg.load(run_id)
    preds = model.predict(X_v)
    return VariantReport(
        test_id=test_id,
        variant=variant,
        run_id=run_id,
        segment_year=segment_year,
        n_rows=int(len(X_v)),
        accuracy=float(accuracy_score(y_v, preds)),
        macro_f1=float(f1_score(y_v, preds, average="macro")),
    )


def _cli() -> int:
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(
        description="Score one A/B variant on its half of the unseen segment. "
                    "Invoked by ab_flow.py inside the ab container.",
    )
    p.add_argument("--run-id", required=True, help="Model version id for this variant.")
    p.add_argument("--variant", required=True, choices=["A", "B", "a", "b"])
    p.add_argument("--test-id", required=True, help="Experiment id (also the split salt).")
    p.add_argument("--segment-year", type=int, default=2024)
    p.add_argument("--out-dir", default=None,
                   help="If set, write variant_<a|b>.json here.")
    args = p.parse_args()
    report = score_variant(args.run_id, args.variant, args.test_id, args.segment_year)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"variant_{report.variant.lower()}.json").write_text(
            json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
