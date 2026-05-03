"""
Tiny file-based model registry.

Each registered model lives at ``models/v<N>/`` with three files:
  - ``model.skops``   -- the serialized sklearn pipeline (skops, not pickle)
  - ``schema.json``   -- input/output schema + code dependencies + git_sha
  - ``run.json``      -- robustness report (written after evaluation)

A flat ``models/INDEX.json`` is regenerated on every write so the set of
registered versions and their key metadata is reviewable from a single
diff-friendly file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import skops.io as sio

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
INDEX_PATH = MODELS_DIR / "INDEX.json"


def _versions() -> list[int]:
    if not MODELS_DIR.exists():
        return []
    return sorted(int(p.name[1:]) for p in MODELS_DIR.iterdir()
                  if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit())


def _index_entry(version: int) -> dict:
    d = MODELS_DIR / f"v{version}"
    sch = json.loads((d / "schema.json").read_text())
    entry = {
        "version": version,
        "path": f"models/v{version}",
        "created_at": sch.get("created_at"),
        "git_sha": sch.get("git_sha"),
        "train_years": sch.get("train_years"),
        "train_accuracy": sch.get("train_accuracy"),
        "classes": sch.get("output", {}).get("classes"),
        "requirements_file": sch.get("code_dependencies", {}).get("requirements_file"),
    }
    run_path = d / "run.json"
    if run_path.exists():
        entry["run"] = json.loads(run_path.read_text())
    return entry


def _write_index() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps([_index_entry(v) for v in _versions()], indent=2))


def save(model, schema: dict) -> int:
    """Persist a model + schema as a new version. Returns the version number."""
    MODELS_DIR.mkdir(exist_ok=True)
    version = (_versions()[-1] + 1) if _versions() else 1
    d = MODELS_DIR / f"v{version}"
    d.mkdir()
    schema = {**schema, "created_at": datetime.now(timezone.utc).isoformat()}
    sio.dump(model, d / "model.skops")
    (d / "schema.json").write_text(json.dumps(schema, indent=2))
    _write_index()
    return version


def attach_run(version: int, run: dict) -> None:
    """Persist a run report (e.g. robustness output) next to the model."""
    d = MODELS_DIR / f"v{version}"
    if not d.exists():
        raise FileNotFoundError(f"no registered model at {d}")
    (d / "run.json").write_text(json.dumps(run, indent=2))
    _write_index()


def list_models() -> list[dict]:
    """Return a list of {version, path, schema} for every registered model."""
    out = []
    for v in _versions():
        d = MODELS_DIR / f"v{v}"
        out.append({"version": v, "path": str(d), "schema": json.loads((d / "schema.json").read_text())})
    return out


def load(version: int):
    p = MODELS_DIR / f"v{version}" / "model.skops"
    return sio.load(p, trusted=sio.get_untrusted_types(file=p))


def schema(version: int) -> dict:
    return json.loads((MODELS_DIR / f"v{version}" / "schema.json").read_text())


def latest_version() -> int:
    vs = _versions()
    if not vs:
        raise RuntimeError("no registered models; train one first")
    return vs[-1]
