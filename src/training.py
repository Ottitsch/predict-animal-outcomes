"""
Train a small multinomial logistic-regression classifier and register it.

The model is intentionally tiny: a one-shot ``LogisticRegression.fit`` over a
one-hot-encoded feature matrix. The point is to produce an artifact we can
version, load, and validate.

Error handling
==============
Two failure modes are explicitly handled:

* **Too-small training set** (``< 1000`` rows) raises ``ValueError`` before any
  artifact is written. Refusing to produce a junk model is strictly better than
  emitting one that could be promoted by accident: a missing model is a loud
  failure, an under-trained registered model is a silent one.
* **Mid-training interruption** is simulated by ``simulate_interrupt=True``,
  which raises ``RuntimeError`` after data load but before fitting. The flow
  step propagates the error and the model is never registered. Recovery is a
  plain re-run -- there is no checkpoint/resume because a single one-shot
  ``fit`` has nothing meaningful to checkpoint.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data as data_mod
from . import registry as reg


@dataclass
class TrainResult:
    run_id: str
    train_accuracy: float



def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), data_mod.CATEGORICAL_FEATURES),
            ("num", StandardScaler(), data_mod.NUMERIC_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
    ])


def train(train_years: list[int], run_id: str, simulate_interrupt: bool = False) -> TrainResult:
    df = data_mod.load_years(train_years)
    X, y = data_mod.split_xy(df)

    if len(X) < 1000:
        raise ValueError(
            f"refusing to train on only {len(X)} rows (<1000); "
            f"check that train_years={train_years} resolved to real data"
        )
    if simulate_interrupt:
        raise RuntimeError("simulated training interrupt (set simulate_interrupt=False to train)")

    pipe = _build_pipeline()
    pipe.fit(X, y)
    acc = float(pipe.score(X, y))

    schema = {
        "input": {
            "categorical_features": data_mod.CATEGORICAL_FEATURES,
            "numeric_features": data_mod.NUMERIC_FEATURES,
            "feature_columns": data_mod.FEATURE_COLUMNS,
        },
        "output": {
            "target_column": data_mod.TARGET_COLUMN,
            "classes": [str(c) for c in pipe.classes_],
            "type": "categorical",
        },
        "code_dependencies": {
            "python": platform.python_version(),
            "requirements_file": "docker/requirements/train.txt",
        },
        "git_sha": run_id.split("__")[-1],
        "train_years": train_years,
        "train_accuracy": acc,
    }
    reg.save(pipe, schema, run_id)
    print(f"[train] registered model for run {run_id} train_acc={acc:.4f}")
    return TrainResult(run_id=run_id, train_accuracy=acc)


def _cli() -> int:
    import argparse, json
    p = argparse.ArgumentParser(
        description="Train + register the model. Invoked by flow.py inside the "
                    "train container; not intended for ad-hoc use.",
    )
    p.add_argument("--run-id", required=True,
                   help="Run id this training belongs to. Required: every model "
                        "is colocated with its run at runs/<run-id>/model/.")
    p.add_argument("--years", default="2014,2015,2016,2017,2018,2019,2020,2021,2022")
    p.add_argument("--simulate-interrupt", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="If set, write a per-step summary.json here (used by "
                        "the flow runner to read back train_accuracy).")
    args = p.parse_args()
    years = [int(y) for y in args.years.split(",")]
    res = train(train_years=years, run_id=args.run_id,
                simulate_interrupt=args.simulate_interrupt)
    summary = {"run_id": res.run_id, "train_accuracy": res.train_accuracy}
    print(json.dumps(summary, indent=2))
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
