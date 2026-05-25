"""
File-based model registry rooted in run directories.

Each successful flow run produces exactly one model, persisted at
``runs/<run-id>/model/`` with two files:

  - ``model.skops``   -- the serialized sklearn pipeline (skops, not pickle)
  - ``schema.json``   -- input/output schema + code dependencies + git_sha
                         + the originating ``run_id``

Models are identified by their producing ``run_id`` (which is timestamped and
git-sha-suffixed), not by a separate version namespace. The robustness report
for a run lives at ``runs/<run-id>/robustness/report.json`` -- not duplicated
into the model dir.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def _model_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id / "model"


def save(model, schema: dict, run_id: str) -> str:
    """Persist a model + schema under the given run. Returns the run_id."""
    import skops.io as sio

    d = _model_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    schema = {**schema, "run_id": run_id,
              "created_at": datetime.now(timezone.utc).isoformat()}
    sio.dump(model, d / "model.skops")
    (d / "schema.json").write_text(json.dumps(schema, indent=2))
    return run_id


def load(run_id: str):
    # Imported lazily so the file-listing/schema helpers below (used by the
    # flow drivers, which run in a minimal image without skops) don't pull in
    # the serialization stack just to resolve a run id.
    import skops.io as sio

    p = _model_dir(run_id) / "model.skops"
    return sio.load(p, trusted=sio.get_untrusted_types(file=p))


def schema(run_id: str) -> dict:
    return json.loads((_model_dir(run_id) / "schema.json").read_text())


def list_runs_with_model() -> list[str]:
    """Run ids, oldest-first, that have a registered model on disk."""
    if not RUNS_DIR.exists():
        return []
    out = [p.name for p in RUNS_DIR.iterdir()
           if p.is_dir() and (p / "model" / "model.skops").exists()]
    return sorted(out)


def latest_run_id() -> str:
    runs = list_runs_with_model()
    if not runs:
        raise RuntimeError("no registered models; train one first")
    return runs[-1]
