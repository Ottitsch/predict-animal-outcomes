"""
Shared helpers for spawning per-step containers from a flow driver.

The training flow (flow.py) predates this module and keeps its own copy of the
same logic; the monitoring and A/B flows use these helpers so the
sibling-container spawning contract lives in one place. See flow.py's module
docstring for the full description of the bind-mount / blocking-exit-code model;
the short version:

  * ``$HOST_PROJECT_DIR``        -> ``/work`` (the repo)
  * ``$HOST_PROJECT_DIR/<sub>``  -> ``/out``  (this step's output dir)

``docker run`` is foreground/blocking, so the subprocess returns exactly when
the step container's PID 1 exits, and its exit code is the step's result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USE_CONTAINERS = os.environ.get("USE_CONTAINERS", "1") != "0"
IMAGE_TAG = os.environ.get("PAO_IMAGE_TAG", "dev")


def image(name: str) -> str:
    return f"pao-{name}:{IMAGE_TAG}"


def detect_host_project_dir() -> str:
    """Host-side path bind-mounted to /work (see flow.py for the rationale)."""
    if not USE_CONTAINERS:
        return str(ROOT)
    try:
        import socket
        cid = socket.gethostname()
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", cid],
            capture_output=True, text=True, check=True,
        )
        for mount in json.loads(result.stdout):
            if mount.get("Destination") == "/work":
                return mount["Source"]
    except Exception:
        pass
    return str(ROOT)


HOST_PROJECT_DIR = detect_host_project_dir()


def docker_run(image_name: str, out_subpath: str, cmd: list[str],
               stdout_log: Path) -> int:
    """Spawn a sibling container, mounting ``runs/<out_subpath>`` as /out.

    Blocks until the container exits; returns its exit code.
    """
    out_host = f"{HOST_PROJECT_DIR}/{out_subpath}"
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{HOST_PROJECT_DIR}:/work",
        "-v", f"{out_host}:/out",
        "-w", "/work",
        image(image_name),
        *cmd,
    ]
    return _stream(docker_cmd, stdout_log, cwd=None)


def host_run(cmd: list[str], stdout_log: Path) -> int:
    """Same contract as docker_run but runs the command on the host directly."""
    return _stream(cmd, stdout_log, cwd=ROOT)


def _stream(cmd: list[str], stdout_log: Path, cwd) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    with stdout_log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=cwd)
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
    return proc.returncode
