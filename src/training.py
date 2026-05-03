"""
Train a "very simple" multinomial logistic-regression classifier on the
per-year cleaned datasets.

The model is intentionally small (SGDClassifier with log loss = streaming
multinomial logistic regression). The size and sophistication of the model is
not the point of this exercise -- we need an artifact that can be versioned,
loaded, and validated.

Error handling
==============
The training step supports an injected interruption to demonstrate handling
of the "system is interrupted during training" scenario.

  * ``--interrupt-at-epoch N`` raises ``KeyboardInterrupt`` after epoch N.
  * On interruption we save the *partial* model to ``checkpoints/<run_id>.skops``
    and tag the corresponding MLflow run with ``status=interrupted`` and
    ``last_completed_epoch``. The run is not registered in the model registry
    (an interrupted run produces an under-trained model and must not silently
    become a deployment candidate).
  * ``--resume-from <ckpt-path>`` reloads the partial model and continues from
    the next epoch. A normal completion finalises and registers the model.

This design treats interruptions like power loss in a long-running training
job: progress is durable, recovery is explicit, and a half-trained model can
never accidentally be promoted to production.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from . import data as data_mod
from . import registry as reg
from .preprocessing import to_string_lists

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_EPOCHS = 8


@dataclass
class TrainResult:
    run_id: str
    model_version: str | None  # populated only on successful completion
    train_accuracy: float
    last_completed_epoch: int
    status: str  # "completed" or "interrupted"
    checkpoint_path: str | None


def _build_preprocessor() -> ColumnTransformer:
    """Categorical hashing for cats, scaling for the numeric column."""
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import Pipeline as SkPipeline

    cat_pipeline = SkPipeline(
        [
            # ``to_string_lists`` lives in ``src.preprocessing`` so that its
            # import path stays stable when training is launched as either
            # ``python -m src.training`` (where the module is ``__main__``)
            # or imported from the flow.
            ("to_strings", FunctionTransformer(to_string_lists, validate=False)),
            ("hasher", FeatureHasher(n_features=512, input_type="string")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, data_mod.CATEGORICAL_FEATURES),
            ("num", StandardScaler(), data_mod.NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def _save_skops(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(obj, path)


def _load_skops(path: Path):
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


def _schema_payload(classes: list[str]) -> dict:
    return {
        "input": {
            "categorical_features": data_mod.CATEGORICAL_FEATURES,
            "numeric_features": data_mod.NUMERIC_FEATURES,
            "feature_columns": data_mod.FEATURE_COLUMNS,
        },
        "output": {
            "target_column": data_mod.TARGET_COLUMN,
            "classes": list(classes),
            "type": "categorical",
        },
        "code_dependencies": {
            "python": platform.python_version(),
            "requirements_file": "requirements/train.txt",
        },
    }


def train(
    train_years: list[int],
    epochs: int = DEFAULT_EPOCHS,
    interrupt_at_epoch: int | None = None,
    resume_from: str | None = None,
) -> TrainResult:
    """Train the model end-to-end with checkpointing + recovery."""
    reg.configure()
    df = data_mod.load_years(train_years)
    X, y = data_mod.split_xy(df)

    if len(X) < 1000:
        # Defensive guard: accidentally training on too little data is a real
        # failure mode and is worth refusing loudly rather than producing a
        # junk artifact that could be promoted by mistake.
        raise ValueError(
            f"refusing to train on only {len(X)} rows (<1000); "
            f"check that train_years={train_years} resolved to real data"
        )

    classes = sorted(y.unique().tolist())

    if resume_from:
        ckpt = _load_skops(Path(resume_from))
        preprocessor = ckpt["preprocessor"]
        clf = ckpt["clf"]
        start_epoch = ckpt["last_completed_epoch"] + 1
        Xt = preprocessor.transform(X)
        print(f"[train] resuming from {resume_from} at epoch {start_epoch}")
    else:
        preprocessor = _build_preprocessor()
        Xt = preprocessor.fit_transform(X)
        clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1)
        start_epoch = 1

    yv = y.values
    classes_arr = np.array(classes)
    rng = np.random.default_rng(42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "model": "SGDClassifier(log_loss)",
                "epochs_target": epochs,
                "train_years": ",".join(map(str, train_years)),
                "n_train_rows": len(X),
                "interrupt_at_epoch": interrupt_at_epoch,
                "resumed_from": resume_from or "",
                "start_epoch": start_epoch,
            }
        )

        try:
            for epoch in range(start_epoch, epochs + 1):
                idx = rng.permutation(len(yv))
                clf.partial_fit(Xt[idx], yv[idx], classes=classes_arr)
                acc = float((clf.predict(Xt) == yv).mean())
                mlflow.log_metric("train_accuracy", acc, step=epoch)
                print(f"[train] epoch {epoch}/{epochs} train_acc={acc:.4f}")

                if interrupt_at_epoch is not None and epoch == interrupt_at_epoch:
                    raise KeyboardInterrupt(f"injected interrupt at epoch {epoch}")

        except KeyboardInterrupt as exc:
            # Treat the in-flight epoch as not fully completed: the injected
            # interrupt fires after partial_fit but before we'd have started
            # epoch N+1, and a real SIGINT could land anywhere inside the loop.
            last_done = epoch - 1
            ckpt_path = CHECKPOINT_DIR / f"{run_id}.skops"
            _save_skops(
                {
                    "preprocessor": preprocessor,
                    "clf": clf,
                    "last_completed_epoch": last_done,
                    "classes": classes,
                },
                ckpt_path,
            )
            mlflow.log_metric("last_completed_epoch", last_done)
            mlflow.set_tag("status", "interrupted")
            mlflow.set_tag("checkpoint_path", str(ckpt_path))
            mlflow.set_tag("interruption_reason", str(exc))
            print(
                f"[train] INTERRUPTED at epoch {epoch}: {exc}\n"
                f"[train] checkpoint saved to {ckpt_path}\n"
                f"[train] resume with: --resume-from {ckpt_path}"
            )
            return TrainResult(
                run_id=run_id,
                model_version=None,
                train_accuracy=acc if "acc" in locals() else float("nan"),
                last_completed_epoch=last_done,
                status="interrupted",
                checkpoint_path=str(ckpt_path),
            )

        # Normal completion: bundle preprocessor + classifier as a single
        # callable estimator so consumers don't have to know about both pieces.
        from sklearn.pipeline import Pipeline as SkPipeline
        import tempfile
        from .pyfunc_model import AnimalOutcomePyfunc

        bundle = SkPipeline([("pre", preprocessor), ("clf", clf)])

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            model_path = tmp / "model.skops"
            schema_path = tmp / "schema.json"
            _save_skops(bundle, model_path)
            with open(schema_path, "w") as f:
                json.dump(_schema_payload(classes), f, indent=2)
            deps_src = ROOT / "requirements" / "train.txt"
            deps_tmp = tmp / "train.txt"
            if deps_src.exists():
                deps_tmp.write_bytes(deps_src.read_bytes())

            # Log a proper MLflow logged_model (PyFunc flavour) so the model
            # lands in the Model Registry with a real version. The skops file
            # is referenced as an artifact of the logged_model and is the
            # actual on-disk format -- the PythonModel wrapper just defers to
            # ``skops.io.load`` at predict time.
            artifacts = {"skops_model": str(model_path)}
            if deps_src.exists():
                artifacts["dependencies"] = str(deps_tmp)
            artifacts["schema"] = str(schema_path)

            logged = mlflow.pyfunc.log_model(
                name="model",
                python_model=AnimalOutcomePyfunc(),
                artifacts=artifacts,
                pip_requirements=str(deps_src) if deps_src.exists() else None,
                registered_model_name=reg.MODEL_NAME,
            )

        mlflow.set_tag("status", "completed")
        mlflow.set_tag("model_format", "skops")

        # Look up the freshly-registered version so callers know what to load.
        client = reg.configure()
        versions = client.search_model_versions(
            f"name='{reg.MODEL_NAME}' and run_id='{run_id}'"
        )
        version = versions[0].version if versions else None

        print(f"[train] completed: run_id={run_id} version={version} acc={acc:.4f}")
        return TrainResult(
            run_id=run_id,
            model_version=version,
            train_accuracy=acc,
            last_completed_epoch=epochs,
            status="completed",
            checkpoint_path=None,
        )


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="2014,2015,2016,2017,2018,2019,2020,2021,2022")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--interrupt-at-epoch", type=int, default=None)
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()
    years = [int(y) for y in args.years.split(",")]
    res = train(
        train_years=years,
        epochs=args.epochs,
        interrupt_at_epoch=args.interrupt_at_epoch,
        resume_from=args.resume_from,
    )
    print(json.dumps(res.__dict__, indent=2))
    return 0 if res.status == "completed" else 2


if __name__ == "__main__":
    sys.exit(_cli())
