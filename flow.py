"""
Metaflow flow: data tests -> training -> robustness validation.

Architecture
------------
The flow is the *driver*. Each step shells out to a step-specific Docker image
that contains only that step's dependencies (see ``docker/*.Dockerfile`` and
``requirements/*.txt``). Inputs and outputs are passed via two bind mounts:

* ``$HOST_PROJECT_DIR``  -> ``/work``        (the repo; data + models registry)
* ``$HOST_PROJECT_DIR/runs/<run-id>/<step>`` -> ``/out`` (this step's outputs)

When the host itself runs inside a container (the ``pao-host`` image),
the bind-mount paths we pass to ``docker run`` are interpreted by the host
daemon, not by anything inside our container. ``_detect_host_project_dir``
resolves this automatically by inspecting the running container's mounts via
the Docker socket, so no ``HOST_PROJECT_DIR`` env var is needed.

How the host knows when a step finishes
---------------------------------------
``docker run`` (foreground) is blocking. ``subprocess.run([..., "docker", "run"
, ...], check=True)`` returns when the step container's PID 1 exits and
propagates the exit code. Non-zero exit -> CalledProcessError -> Metaflow stops
the flow. No polling, no healthchecks, no signals.

Run locally:
    docker/build.sh
    docker run --rm \\
        -v /var/run/docker.sock:/var/run/docker.sock \\
        -v "$PWD":/work -w /work \\
        pao-host:dev

Inject a training error to demonstrate the failure path:
    pao-host:dev run --simulate_interrupt True

Run without containers (legacy, host-native):
    USE_CONTAINERS=0 python flow.py run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from metaflow import FlowSpec, Parameter, step

ROOT = Path(__file__).resolve().parent
USE_CONTAINERS = os.environ.get("USE_CONTAINERS", "1") != "0"

_cfg_path = ROOT / "config.yml"
_cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}


def _detect_host_project_dir() -> str:
    """Return the host-side path that is bind-mounted to /work.

    When pao-host runs inside Docker, bind-mount paths passed to sibling
    ``docker run`` calls are resolved by the host daemon — so we need the
    *host* path, not /work. We get it by asking the daemon to inspect this
    container (identified by its hostname, which Docker sets to the short
    container ID) and reading the Source of the /work mount.

    Falls back to ROOT when running outside a container or if the socket is
    unavailable (e.g. USE_CONTAINERS=0 host-native mode).
    """
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


HOST_PROJECT_DIR = _detect_host_project_dir()
IMAGE_TAG = os.environ.get("PAO_IMAGE_TAG", "dev")

IMAGES = {
    "data_tests": f"pao-data-tests:{IMAGE_TAG}",
    "train":      f"pao-train:{IMAGE_TAG}",
    "robustness": f"pao-robustness:{IMAGE_TAG}",
}


def _docker_run(step_name: str, run_id: str, cmd: list[str], stdout_log: Path) -> int:
    """Spawn a sibling container for one step and stream its output to a log file.

    Blocks until the container exits. Returns the container's exit code.
    """
    image = IMAGES[step_name]
    out_host = f"{HOST_PROJECT_DIR}/runs/{run_id}/{step_name}"
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{HOST_PROJECT_DIR}:/work",
        "-v", f"{out_host}:/out",
        "-w", "/work",
        image,
        *cmd,
    ]
    print(f"[{step_name}] $ {' '.join(docker_cmd)}", flush=True)
    with stdout_log.open("w") as f:
        proc = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
    return proc.returncode


def _host_run(cmd: list[str], stdout_log: Path) -> int:
    """Same contract as _docker_run but runs the command on the host directly."""
    print(f"[host] $ {' '.join(cmd)}", flush=True)
    with stdout_log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
    return proc.returncode


class AnimalOutcomeFlow(FlowSpec):
    train_years = Parameter(
        "train_years",
        help="Comma-separated list of years to train on.",
        default=_cfg.get("train_years", "2014,2015,2016,2017,2018,2019,2020,2021,2022"),
    )
    holdout_year = Parameter(
        "holdout_year",
        help="Year to use for the robustness evaluation.",
        default=_cfg.get("holdout_year", 2024),
        type=int,
    )
    simulate_interrupt = Parameter(
        "simulate_interrupt",
        help="If True, raise during training to demo the error path.",
        default=False,
        type=bool,
    )
    auto_commit = Parameter(
        "auto_commit",
        help="If True, git-commit the run dir + new model after a successful flow.",
        default=True,
        type=bool,
    )

    @step
    def start(self):
        from src import runs

        self.run_id = runs.new_run_id()
        self.run_path = str(runs.run_dir(self.run_id))
        runs.init_envelope(self.run_id, parameters={
            "train_years": self.train_years,
            "holdout_year": self.holdout_year,
            "simulate_interrupt": self.simulate_interrupt,
            "use_containers": USE_CONTAINERS,
        })
        print(f"=== Animal Outcome Flow ===")
        print(f"  run_id             = {self.run_id}")
        print(f"  use_containers     = {USE_CONTAINERS}")
        print(f"  train_years        = {self.train_years}")
        print(f"  holdout_year       = {self.holdout_year}")
        print(f"  simulate_interrupt = {self.simulate_interrupt}")
        self.next(self.data_tests)

    @step
    def data_tests(self):
        """Run the data quality tests with pytest in an isolated container.

        We do not gate the flow on this step's exit code: the test suite
        intentionally includes one expected failure on the raw dataset (a known
        data-quality issue we want surfaced but not blocking). A real
        containerized test runner could be made strict by raising on non-zero
        returncode here.
        """
        from src import runs

        out_dir = runs.step_dir(self.run_id, "data_tests")
        stdout_log = out_dir / "stdout.log"
        junit = "/out/junit.xml" if USE_CONTAINERS else str(out_dir / "junit.xml")
        cmd = ["python", "-m", "pytest", "tests/", "-q", f"--junitxml={junit}"]
        t0 = time.monotonic()
        if USE_CONTAINERS:
            rc = _docker_run("data_tests", self.run_id, cmd, stdout_log)
        else:
            rc = _host_run(cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        (out_dir / "returncode.txt").write_text(str(rc))
        runs.update_step(self.run_id, "data_tests",
                         status="passed" if rc == 0 else "failed_nonblocking",
                         returncode=rc, duration_s=duration,
                         image=IMAGES["data_tests"] if USE_CONTAINERS else "host")
        self.tests_returncode = rc
        if rc != 0:
            print(f"[data_tests] pytest exit={rc} (non-blocking; see {stdout_log})")
        self.next(self.train)

    @step
    def train(self):
        """Train + register the model in an isolated container.

        Two failure modes are explicitly handled in ``src/training.py`` and
        propagate through the container exit code:

        * **Too-small training set** (``< 1000`` rows) -> exit 1
        * **Simulated mid-training error** (``simulate_interrupt=True``) -> exit 1
        """
        from src import runs

        out_dir = runs.step_dir(self.run_id, "train")
        stdout_log = out_dir / "stdout.log"
        cmd = [
            "python", "-m", "src.training",
            "--years", self.train_years,
            "--out-dir", "/out" if USE_CONTAINERS else str(out_dir),
        ]
        if self.simulate_interrupt:
            cmd.append("--simulate-interrupt")
        t0 = time.monotonic()
        if USE_CONTAINERS:
            rc = _docker_run("train", self.run_id, cmd, stdout_log)
        else:
            rc = _host_run(cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        if rc != 0:
            runs.update_step(self.run_id, "train", status="failed",
                             returncode=rc, duration_s=duration,
                             image=IMAGES["train"] if USE_CONTAINERS else "host")
            runs.finalize(self.run_id, "failed")
            raise RuntimeError(f"train step failed with exit={rc}")
        summary = json.loads((out_dir / "summary.json").read_text())
        self.model_version = summary["version"]
        self.train_accuracy = summary["train_accuracy"]
        runs.update_step(self.run_id, "train", status="passed",
                         returncode=rc, duration_s=duration,
                         image=IMAGES["train"] if USE_CONTAINERS else "host",
                         model_version=self.model_version,
                         train_accuracy=self.train_accuracy)
        self.next(self.robustness)

    @step
    def robustness(self):
        """Validate the freshly trained model against the documented thresholds."""
        from src import runs

        out_dir = runs.step_dir(self.run_id, "robustness")
        stdout_log = out_dir / "stdout.log"
        cmd = [
            "python", "-m", "src.robustness",
            "--holdout-year", str(self.holdout_year),
            "--out-dir", "/out" if USE_CONTAINERS else str(out_dir),
        ]
        t0 = time.monotonic()
        if USE_CONTAINERS:
            rc = _docker_run("robustness", self.run_id, cmd, stdout_log)
        else:
            rc = _host_run(cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        report_path = out_dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        self.robustness_report = report
        status = "passed" if rc == 0 else "failed"
        runs.update_step(self.run_id, "robustness", status=status,
                         returncode=rc, duration_s=duration,
                         image=IMAGES["robustness"] if USE_CONTAINERS else "host",
                         report=report)
        if rc != 0:
            runs.finalize(self.run_id, "failed")
            raise RuntimeError(f"robustness step failed with exit={rc}")
        print(f"[robustness] PASSED  margin={report.get('margin_over_baseline'):.4f}"
              f"  gap={report.get('train_holdout_gap'):.4f}")
        self.next(self.end)

    @step
    def end(self):
        from src import runs
        runs.finalize(self.run_id, "passed",
                      model_version=self.model_version,
                      train_accuracy=self.train_accuracy)
        print(f"=== flow complete ===")
        print(f"  run_id         = {self.run_id}")
        print(f"  model_version  = {self.model_version}")
        print(f"  train_accuracy = {self.train_accuracy:.4f}")
        if self.auto_commit:
            self._auto_commit()

    def _auto_commit(self):
        """Commit the run dir + (any new) model artifacts.

        Lowercase, short message per the project's commit conventions.
        Uses --no-gpg-sign and the configured ottitsch identity.
        """
        run_path = Path(self.run_path).relative_to(ROOT)
        paths = [str(run_path)]
        model_dir = ROOT / "models" / f"v{self.model_version}"
        if model_dir.exists():
            paths.extend([f"models/v{self.model_version}", "models/INDEX.json"])
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "ottitsch",
            "GIT_AUTHOR_EMAIL": "ottitsch.work@gmail.com",
            "GIT_COMMITTER_NAME": "ottitsch",
            "GIT_COMMITTER_EMAIL": "ottitsch.work@gmail.com",
        }
        try:
            subprocess.run(["git", "add", "--", *paths], cwd=ROOT, env=env, check=True)
            msg = f"run {self.run_id.lower()} v{self.model_version} passed"
            subprocess.run(["git", "commit", "--no-gpg-sign", "-m", msg],
                           cwd=ROOT, env=env, check=True)
            print(f"[auto-commit] committed: {msg}")
        except subprocess.CalledProcessError as e:
            print(f"[auto-commit] skipped: {e}")


if __name__ == "__main__":
    AnimalOutcomeFlow()
