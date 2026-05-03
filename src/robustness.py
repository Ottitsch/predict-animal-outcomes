"""
Robustness check for the most recently registered model.

Expectation
-----------
We treat the most recently registered model as the deployment candidate and
require it to clear two bars on a held-out year that it never saw during
training (default: 2024):

  1. **Beat the majority-class baseline by a meaningful margin.**
     A dummy classifier that always predicts the most common outcome
     ("Adoption", ~49%) is the floor below which the model adds no value.
     We require ``model_accuracy >= majority_accuracy + MARGIN``
     with ``MARGIN = 0.05`` (5 percentage points). The choice of 5pp is small
     enough to not be defeated by year-to-year class-mix noise, and large
     enough that a model that merely re-discovers the prior would fail.

  2. **No catastrophic train/holdout gap.**
     We require ``train_accuracy - holdout_accuracy <= MAX_GAP`` with
     ``MAX_GAP = 0.10`` (10 percentage points). A larger gap signals
     overfitting to the training years or distribution drift that the model
     hasn't generalised across, both of which we want to block before
     deployment.

These thresholds catch the failure modes that matter before a model is shipped
(useless models and over-fit/under-generalised models). They are deliberately
loose enough not to be tripped by ordinary year-to-year noise. If the model
fails either check, that should prompt a real investigation, not a threshold
tweak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.dummy import DummyClassifier

from . import data as data_mod
from . import registry as reg

MAJORITY_MARGIN = 0.05
MAX_TRAIN_HOLDOUT_GAP = 0.10


@dataclass
class RobustnessReport:
    model_version: str
    holdout_year: int
    n_holdout_rows: int
    majority_baseline_accuracy: float
    holdout_accuracy: float
    train_accuracy: float
    margin_over_baseline: float
    train_holdout_gap: float
    passed_majority: bool
    passed_gap: bool
    passed: bool


def evaluate(holdout_year: int) -> RobustnessReport:
    model, entry = reg.load_latest_model()

    df = data_mod.load_year(holdout_year)
    X, y = data_mod.split_xy(df)
    if len(X) == 0:
        raise RuntimeError(f"holdout year {holdout_year} has no usable rows")

    holdout_acc = float((model.predict(X) == y.values).mean())

    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    majority_acc = float((dummy.predict(X) == y.values).mean())

    train_acc = float(entry.metrics.get("train_accuracy", float("nan")))
    margin = holdout_acc - majority_acc
    gap = train_acc - holdout_acc

    passed_majority = margin >= MAJORITY_MARGIN
    passed_gap = gap <= MAX_TRAIN_HOLDOUT_GAP
    return RobustnessReport(
        model_version=entry.version,
        holdout_year=holdout_year,
        n_holdout_rows=len(X),
        majority_baseline_accuracy=majority_acc,
        holdout_accuracy=holdout_acc,
        train_accuracy=train_acc,
        margin_over_baseline=margin,
        train_holdout_gap=gap,
        passed_majority=passed_majority,
        passed_gap=passed_gap,
        passed=passed_majority and passed_gap,
    )


def _cli() -> int:
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-year", type=int, default=2024)
    args = p.parse_args()
    report = evaluate(args.holdout_year)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
