"""
Tiny file-based model registry.

Each registered model lives at ``models/v<N>/`` with two files:
  - ``model.skops``   -- the serialized sklearn pipeline (skops, not pickle)
  - ``schema.json``   -- input/output schema + code dependencies
"""
from __future__ import annotations

import json
from pathlib import Path

import skops.io as sio

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _versions() -> list[int]:
    if not MODELS_DIR.exists():
        return []
    return sorted(int(p.name[1:]) for p in MODELS_DIR.iterdir()
                  if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit())


def save(model, schema: dict) -> int:
    """Persist a model + schema as a new version. Returns the version number."""
    MODELS_DIR.mkdir(exist_ok=True)
    version = (_versions()[-1] + 1) if _versions() else 1
    d = MODELS_DIR / f"v{version}"
    d.mkdir()
    sio.dump(model, d / "model.skops")
    (d / "schema.json").write_text(json.dumps(schema, indent=2))
    return version


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
