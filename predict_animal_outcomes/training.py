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



def _build_pipeline(
    *,
    C: float = 1.0,
    class_weight: str | None = None,
    solver: str = "lbfgs",
    categorical_features: list[str] | None = None,
    scale_age: bool = True,
) -> Pipeline:
    """Assemble the training pipeline from configurable hyperparameters.

    Every knob here is driven by a flow parameter (see ``config.yml`` / the
    ``flow.py`` Parameters), so two runs that differ only in these values are
    two distinct, independently reproducible model versions.
    """
    categorical_features = (categorical_features
                            if categorical_features is not None
                            else list(data_mod.CATEGORICAL_FEATURES))
    num_transformer = StandardScaler() if scale_age else "passthrough"
    return Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", num_transformer, data_mod.NUMERIC_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight=class_weight,
                                   solver=solver, n_jobs=None)),
    ])


def train(
    train_years: list[int],
    run_id: str,
    simulate_interrupt: bool = False,
    *,
    model_C: float = 1.0,
    model_class_weight: str | None = None,
    model_solver: str = "lbfgs",
    drop_features: list[str] | None = None,
    scale_age: bool = True,
) -> TrainResult:
    df = data_mod.load_years(train_years)
    X, y = data_mod.split_xy(df)

    if len(X) < 1000:
        raise ValueError(
            f"refusing to train on only {len(X)} rows (<1000); "
            f"check that train_years={train_years} resolved to real data"
        )
    if simulate_interrupt:
        raise RuntimeError("simulated training interrupt (set simulate_interrupt=False to train)")

    drop_features = drop_features or []
    categorical_features = [c for c in data_mod.CATEGORICAL_FEATURES
                            if c not in drop_features]
    feature_columns = categorical_features + list(data_mod.NUMERIC_FEATURES)

    pipe = _build_pipeline(
        C=model_C, class_weight=model_class_weight, solver=model_solver,
        categorical_features=categorical_features, scale_age=scale_age,
    )
    pipe.fit(X, y)
    acc = float(pipe.score(X, y))

    # Reference distribution for prediction-drift monitoring: the class mix the
    # model itself produced on the data it was trained on. The monitor compares
    # a later segment's predicted-class mix against this baseline (see
    # predict_animal_outcomes/monitoring.py for why this is the chosen "expected" behaviour).
    train_pred = pipe.predict(X)
    classes = [str(c) for c in pipe.classes_]
    counts = {c: 0 for c in classes}
    for p in train_pred:
        counts[str(p)] += 1
    total = float(len(train_pred))
    train_pred_dist = {c: counts[c] / total for c in classes}

    schema = {
        "input": {
            "categorical_features": categorical_features,
            "numeric_features": data_mod.NUMERIC_FEATURES,
            "feature_columns": feature_columns,
        },
        "output": {
            "target_column": data_mod.TARGET_COLUMN,
            "classes": classes,
            "type": "categorical",
        },
        "hyperparameters": {
            "model_C": model_C,
            "model_class_weight": model_class_weight,
            "model_solver": model_solver,
            "drop_features": drop_features,
            "scale_age": scale_age,
            "max_iter": 1000,
        },
        "code_dependencies": {
            "python": platform.python_version(),
            "requirements_file": "docker/requirements/train.txt",
        },
        "git_sha": run_id.split("__")[-1],
        "train_years": train_years,
        "train_accuracy": acc,
        "train_prediction_distribution": train_pred_dist,
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
    p.add_argument("--model-c", type=float, default=1.0)
    p.add_argument("--model-class-weight", default="",
                   help='"" -> None; "balanced" -> reweight by class frequency.')
    p.add_argument("--model-solver", default="lbfgs")
    p.add_argument("--drop-features", default="",
                   help="Comma-separated categorical features to exclude.")
    p.add_argument("--scale-age", default="true",
                   help='"true"/"false": scale age_days or pass it through.')
    p.add_argument("--out-dir", default=None,
                   help="If set, write a per-step summary.json here (used by "
                        "the flow runner to read back train_accuracy).")
    args = p.parse_args()
    years = [int(y) for y in args.years.split(",")]
    class_weight = args.model_class_weight or None
    drop = [c.strip() for c in args.drop_features.split(",") if c.strip()]
    scale_age = str(args.scale_age).lower() not in ("false", "0", "no")
    res = train(train_years=years, run_id=args.run_id,
                simulate_interrupt=args.simulate_interrupt,
                model_C=args.model_c, model_class_weight=class_weight,
                model_solver=args.model_solver, drop_features=drop,
                scale_age=scale_age)
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
