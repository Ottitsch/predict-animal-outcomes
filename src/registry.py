"""
Wrapper around MLflow's local file-store + Model Registry.

The registry supports three operations:
  (a) listing available models and their locations,
  (b) loading a particular model by id, and
  (c) returning each model's input/output schema and code dependencies.

We use MLflow's first-class abstractions for all three:
  - The training step calls ``mlflow.pyfunc.log_model(..., registered_model_name=MODEL_NAME)``
    which produces a *logged_model* and registers a new version of
    ``MODEL_NAME`` in the Model Registry.
  - Each registered version points at artifacts that include ``schema.json``
    (input/output contract) and ``dependencies`` (the pinned ``train.txt``).
  - "Model id" in this module is the registered model version number.

We deliberately stick with the file-store backend (``mlruns/``) so the flow
runs without any external services. MLflow's deprecation warning about the
file backend is acknowledged but does not affect correctness.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "predict-animal-outcomes"
MODEL_NAME = "animal_outcome_classifier"


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_DIR.as_uri())


def configure() -> MlflowClient:
    """Set the tracking URI + experiment, return an MlflowClient."""
    DEFAULT_TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MlflowClient()


@dataclass
class ModelEntry:
    name: str
    version: str
    run_id: str
    source_uri: str  # the artifact location (where MLflow stored the logged_model)
    metrics: dict
    tags: dict
    params: dict


def _entry(client: MlflowClient, mv) -> ModelEntry:
    run = client.get_run(mv.run_id)
    return ModelEntry(
        name=mv.name,
        version=str(mv.version),
        run_id=mv.run_id,
        source_uri=mv.source,
        metrics=dict(run.data.metrics),
        tags=dict(run.data.tags),
        params=dict(run.data.params),
    )


def list_models() -> list[ModelEntry]:
    """List every registered version of the project's model, oldest first."""
    client = configure()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        return []
    return sorted([_entry(client, v) for v in versions], key=lambda e: int(e.version))


def get_entry(version: str | int) -> ModelEntry:
    client = configure()
    return _entry(client, client.get_model_version(MODEL_NAME, str(version)))


def load_model_by_version(version: str | int):
    """Load a registered model version via MLflow's pyfunc loader."""
    configure()
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")


def load_latest_model() -> tuple[object, ModelEntry]:
    entries = list_models()
    if not entries:
        raise RuntimeError("no registered model versions found; train a model first")
    latest = entries[-1]
    return load_model_by_version(latest.version), latest


def load_schema(version: str | int) -> dict:
    """Return the input/output schema + code dependencies for a model version."""
    entry = get_entry(version)
    local_dir = mlflow.artifacts.download_artifacts(entry.source_uri)
    with open(Path(local_dir) / "artifacts" / "schema.json") as f:
        return json.load(f)
