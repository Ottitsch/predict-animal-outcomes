"""
Per-flow-execution run directory + audit envelope.

Each invocation of ``flow.py`` creates a directory under ``runs/<run-id>/``
that captures every step's outputs (stdout, junit reports, schema copies,
robustness reports) plus a top-level ``flow.json`` envelope.

The envelope is written incrementally after each step finishes so that a
crashed run still leaves a partial audit trail on disk.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


def _short_git_sha() -> str:
    # In CI, the full SHA is injected via env var to avoid git safe.directory
    # errors when the container runs as a different user than the workspace owner.
    sha_env = os.environ.get("GIT_COMMIT_SHA", "")
    if sha_env:
        return sha_env[:7]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}__{_short_git_sha()}"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def step_dir(run_id: str, step: str) -> Path:
    d = run_dir(run_id) / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_envelope(run_id: str, parameters: dict) -> Path:
    """Create the run dir and write the initial flow.json."""
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_id": run_id,
        "git_sha": _short_git_sha(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "status": "running",
        "parameters": parameters,
        "steps": {},
    }
    path = d / "flow.json"
    path.write_text(json.dumps(envelope, indent=2))
    return path


def update_step(run_id: str, step: str, **fields) -> None:
    """Patch the envelope with a step's result (status, duration, image, etc.)."""
    path = run_dir(run_id) / "flow.json"
    env = json.loads(path.read_text())
    env["steps"][step] = {**env["steps"].get(step, {}), **fields}
    path.write_text(json.dumps(env, indent=2))


def finalize(run_id: str, status: str, **extra) -> None:
    """Mark the envelope finished; status is 'passed' or 'failed'."""
    path = run_dir(run_id) / "flow.json"
    env = json.loads(path.read_text())
    env["ended_at"] = datetime.now(timezone.utc).isoformat()
    env["status"] = status
    env.update(extra)
    path.write_text(json.dumps(env, indent=2))
