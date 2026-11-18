"""
Robustness check for the most recently registered model.

Expectation
-----------
We treat the most recently registered model as the deployment candidate and
require it to clear two bars on a held-out year that it never saw during
training (default: 2024):

  1. **Beat the majority-class baseline by a meaningful margin.**
     A dummy classifier that always predicts the holdout year's most common
     outcome ("Adoption", 62% of 2024 stays) is the floor below which the model
     adds no value. Fitting it on the holdout itself makes it the strictest
     version of that baseline.
     We require ``model_accuracy >= majority_accuracy + MARGIN``
     with ``MARGIN = 0.05`` (5 percentage points). The choice of 5pp is small
     enough to not be defeated by year-to-year class-mix noise, and large
     enough that a model that merely re-discovers the prior would fail.

  2. **No catastrophic train/holdout gap, in either direction.**
     We require ``|train_accuracy - holdout_accuracy| <= MAX_GAP`` with
     ``MAX_GAP = 0.10`` (10 percentage points). Train far above holdout means
     overfitting to the training years. Holdout far *above* train is just as
     much a warning: the holdout year looks different from the training years
     (the adoption share rose from 38% of 2014 stays to 62% of 2024 stays), so
     accuracy on it says little about the next year. The first version only
     checked one direction and passed a model whose holdout beat train by 10.6
     points.

Accuracy alone hides the rare outcomes, so the report also carries macro F1
(model and baseline) and the recall of every class. They are reported, not
gated: most outcomes other than Adoption, Transfer, Return to Owner and
Euthanasia have a few dozen rows a year, and no simple model recalls them.

These thresholds catch the failure modes that matter before a model is shipped
(useless models and over-fit/under-generalised models). They are deliberately
loose enough not to be tripped by ordinary year-to-year noise. If the model
fails either check, that should prompt a real investigation, not a threshold
tweak.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, recall_score

from . import data as data_mod
from . import registry as reg

MAJORITY_MARGIN = 0.05
MAX_TRAIN_HOLDOUT_GAP = 0.10


@dataclass
class RobustnessReport:
    run_id: str
    holdout_year: int
    n_holdout_rows: int
    majority_baseline_accuracy: float
    holdout_accuracy: float
    train_accuracy: float
    margin_over_baseline: float
    train_holdout_gap: float
    holdout_macro_f1: float
    majority_baseline_macro_f1: float
    holdout_class_share: dict
    holdout_recall_per_class: dict
    passed_majority: bool
    passed_gap: bool
    passed: bool


def evaluate(holdout_year: int, run_id: str) -> RobustnessReport:
    model = reg.load(run_id)
    schema = reg.schema(run_id)

    df = data_mod.load_year(holdout_year)
    X, y = data_mod.split_xy(df)
    if len(X) == 0:
        raise RuntimeError(f"holdout year {holdout_year} has no usable rows")

    preds = model.predict(X)
    holdout_acc = float((preds == y.values).mean())

    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    dummy_preds = dummy.predict(X)
    majority_acc = float((dummy_preds == y.values).mean())

    classes = sorted(y.unique())
    recalls = recall_score(y, preds, labels=classes, average=None, zero_division=0)

    train_acc = float(schema.get("train_accuracy", float("nan")))
    margin = holdout_acc - majority_acc
    gap = train_acc - holdout_acc

    passed_majority = margin >= MAJORITY_MARGIN
    passed_gap = abs(gap) <= MAX_TRAIN_HOLDOUT_GAP
    report = RobustnessReport(
        run_id=run_id,
        holdout_year=holdout_year,
        n_holdout_rows=len(X),
        majority_baseline_accuracy=majority_acc,
        holdout_accuracy=holdout_acc,
        train_accuracy=train_acc,
        margin_over_baseline=margin,
        train_holdout_gap=gap,
        holdout_macro_f1=float(f1_score(y, preds, average="macro", zero_division=0)),
        majority_baseline_macro_f1=float(f1_score(y, dummy_preds, average="macro", zero_division=0)),
        holdout_class_share={c: float(v) for c, v in y.value_counts(normalize=True).items()},
        holdout_recall_per_class={c: float(r) for c, r in zip(classes, recalls)},
        passed_majority=passed_majority,
        passed_gap=passed_gap,
        passed=passed_majority and passed_gap,
    )
    return report


def _cli() -> int:
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(
        description="Evaluate the model produced by a given run. Invoked by "
                    "flow.py inside the robustness container.",
    )
    p.add_argument("--run-id", required=True,
                   help="Run id whose model should be evaluated.")
    p.add_argument("--holdout-year", type=int, default=2024)
    p.add_argument("--out-dir", default=None,
                   help="If set, write the robustness report JSON here.")
    args = p.parse_args()
    report = evaluate(args.holdout_year, args.run_id)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(payload, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
