"""
Metaflow flow: data tests -> training -> robustness validation.

Architecture
------------
The flow is the *driver*. Each step shells out to a step-specific Docker image
that contains only that step's dependencies (see ``docker/*.Dockerfile`` and
``docker/requirements/*.txt``). Inputs and outputs are passed via two bind mounts:

* ``$HOST_PROJECT_DIR``  -> ``/work``        (the repo; data + per-run model dirs)
* ``$HOST_PROJECT_DIR/runs/<run-id>/<step>`` -> ``/out`` (this step's outputs)

When the host itself runs inside a container (the ``pao-host`` image),
the bind-mount paths we pass to ``docker run`` are interpreted by the host
daemon, not by anything inside our container. ``_detect_host_project_dir``
resolves this automatically by inspecting the running container's mounts via
the Docker socket, so no ``HOST_PROJECT_DIR`` env var is needed.

How the host knows when a step finishes
---------------------------------------
``docker run`` (foreground) is blocking. Each step is spawned with
``subprocess.Popen`` (see predict_animal_outcomes/containers.py), its output is
streamed into the step's ``stdout.log``, and the call returns the container's
exit code once its PID 1 exits. A non-zero code makes the step raise, which
stops the flow. No polling, no healthchecks, no signals.

Run locally:
    docker/build.sh
    docker run --rm \\
        -v /var/run/docker.sock:/var/run/docker.sock \\
        -v "$PWD":/work -w /work \\
        pao-host:dev

Inject a training error to demonstrate the failure path:
    pao-host:dev run --simulate_interrupt True

Run without containers (legacy, host-native):
    USE_CONTAINERS=0 python flows/flow.py run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# This driver lives in flows/; the repo root (its parent) holds runs/ and the
# predict_animal_outcomes package. Put it on sys.path so `from
# predict_animal_outcomes import ...` resolves whether launched as `python
# flows/flow.py`, host-native, or in Metaflow's step subprocesses -- in all of
# which sys.path[0] is flows/, not the repo root.
FLOWS_DIR = Path(__file__).resolve().parent
ROOT = FLOWS_DIR.parent
sys.path.insert(0, str(ROOT))

import yaml
from metaflow import FlowSpec, Parameter, step

from predict_animal_outcomes import containers

USE_CONTAINERS = containers.USE_CONTAINERS

# training_defaults.yml sits next to this driver in flows/.
_cfg_path = FLOWS_DIR / "training_defaults.yml"
_cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}


# Step name -> its pao-<name> image. The spawning itself lives in
# predict_animal_outcomes/containers.py, shared with the monitoring and A/B flows.
STEP_IMAGES = {"data_tests": "data-tests", "train": "train", "robustness": "robustness"}


def _run_step(step_name: str, run_id: str, cmd: list[str], stdout_log: Path) -> int:
    """Run one step in its own container (or on the host); return its exit code."""
    if USE_CONTAINERS:
        return containers.docker_run(STEP_IMAGES[step_name], f"runs/{run_id}/{step_name}",
                                     cmd, stdout_log)
    return containers.host_run(cmd, stdout_log)


def _image(step_name: str) -> str:
    return containers.image(STEP_IMAGES[step_name]) if USE_CONTAINERS else "host"


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
    # --- Model + transform hyperparameters (the knobs that define a version) ---
    model_C = Parameter(
        "model_C", help="Inverse regularisation strength for LogisticRegression.",
        default=_cfg.get("model_C", 1.0), type=float,
    )
    model_class_weight = Parameter(
        "model_class_weight",
        help='"" -> None; "balanced" -> reweight classes by frequency.',
        default=str(_cfg.get("model_class_weight", "")),
    )
    model_solver = Parameter(
        "model_solver", help="LogisticRegression solver.",
        default=_cfg.get("model_solver", "lbfgs"),
    )
    drop_features = Parameter(
        "drop_features", help="Comma-separated categorical features to exclude.",
        default=str(_cfg.get("drop_features", "")),
    )
    scale_age = Parameter(
        "scale_age", help="If True, scale age_days; otherwise pass it through.",
        default=bool(_cfg.get("scale_age", True)), type=bool,
    )

    @step
    def start(self):
        from predict_animal_outcomes import runs

        self.run_id = runs.new_run_id()
        self.run_path = str(runs.run_dir(self.run_id))
        runs.init_envelope(self.run_id, parameters={
            "train_years": self.train_years,
            "holdout_year": self.holdout_year,
            "simulate_interrupt": self.simulate_interrupt,
            "use_containers": USE_CONTAINERS,
            "model_C": self.model_C,
            "model_class_weight": self.model_class_weight,
            "model_solver": self.model_solver,
            "drop_features": self.drop_features,
            "scale_age": self.scale_age,
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

        The tests check the exact per year stay files that training and the
        robustness step read (the run's train years plus its holdout year). A
        failing test stops the flow before a model is trained on bad data.
        """
        from predict_animal_outcomes import runs

        out_dir = runs.step_dir(self.run_id, "data_tests")
        stdout_log = out_dir / "stdout.log"
        report = "/out/report.json" if USE_CONTAINERS else str(out_dir / "report.json")
        cmd = [
            "python", "-m", "pytest", "tests/", "--ignore=tests/unit", "-q", "--json-report", f"--json-report-file={report}", "--json-report-indent=2",
            f"--train-years={self.train_years}",
            f"--holdout-year={self.holdout_year}",
        ]
        t0 = time.monotonic()
        rc = _run_step("data_tests", self.run_id, cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        (out_dir / "returncode.txt").write_text(str(rc))
        runs.update_step(self.run_id, "data_tests",
                         status="passed" if rc == 0 else "failed",
                         returncode=rc, duration_s=duration,
                         image=_image("data_tests"))
        self.tests_returncode = rc
        if rc != 0:
            runs.finalize(self.run_id, "failed")
            raise RuntimeError(f"data tests failed with exit={rc}; see {stdout_log}")
        self.next(self.train)

    @step
    def train(self):
        """Train + register the model in an isolated container.

        Two failure modes are explicitly handled in ``predict_animal_outcomes/training.py`` and
        propagate through the container exit code:

        * **Too-small training set** (``< 1000`` rows) -> exit 1
        * **Simulated mid-training error** (``simulate_interrupt=True``) -> exit 1
        """
        from predict_animal_outcomes import runs

        out_dir = runs.step_dir(self.run_id, "train")
        stdout_log = out_dir / "stdout.log"
        cmd = [
            "python", "-m", "predict_animal_outcomes.training",
            "--run-id", self.run_id,
            "--years", self.train_years,
            "--model-c", str(self.model_C),
            "--model-class-weight", self.model_class_weight,
            "--model-solver", self.model_solver,
            "--drop-features", self.drop_features,
            "--scale-age", "true" if self.scale_age else "false",
            "--out-dir", "/out" if USE_CONTAINERS else str(out_dir),
        ]
        if self.simulate_interrupt:
            cmd.append("--simulate-interrupt")
        t0 = time.monotonic()
        rc = _run_step("train", self.run_id, cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        if rc != 0:
            runs.update_step(self.run_id, "train", status="failed",
                             returncode=rc, duration_s=duration,
                             image=_image("train"))
            runs.finalize(self.run_id, "failed")
            raise RuntimeError(f"train step failed with exit={rc}")
        summary = json.loads((out_dir / "summary.json").read_text())
        self.train_accuracy = summary["train_accuracy"]
        runs.update_step(self.run_id, "train", status="passed",
                         returncode=rc, duration_s=duration,
                         image=_image("train"),
                         train_accuracy=self.train_accuracy)
        self.next(self.robustness)

    @step
    def robustness(self):
        """Validate the freshly trained model against the documented thresholds."""
        from predict_animal_outcomes import runs

        out_dir = runs.step_dir(self.run_id, "robustness")
        stdout_log = out_dir / "stdout.log"
        cmd = [
            "python", "-m", "predict_animal_outcomes.robustness",
            "--run-id", self.run_id,
            "--holdout-year", str(self.holdout_year),
            "--out-dir", "/out" if USE_CONTAINERS else str(out_dir),
        ]
        t0 = time.monotonic()
        rc = _run_step("robustness", self.run_id, cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        report_path = out_dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        self.robustness_report = report
        status = "passed" if rc == 0 else "failed"
        runs.update_step(self.run_id, "robustness", status=status,
                         returncode=rc, duration_s=duration,
                         image=_image("robustness"),
                         report=report)
        if rc != 0:
            runs.finalize(self.run_id, "failed")
            raise RuntimeError(f"robustness step failed with exit={rc}")
        print(f"[robustness] PASSED  margin={report.get('margin_over_baseline'):.4f}"
              f"  gap={report.get('train_holdout_gap'):.4f}")
        self.next(self.end)

    @step
    def end(self):
        from predict_animal_outcomes import runs
        runs.finalize(self.run_id, "passed",
                      train_accuracy=self.train_accuracy)
        print(f"=== flow complete ===")
        print(f"  run_id         = {self.run_id}")
        print(f"  train_accuracy = {self.train_accuracy:.4f}")
        if self.auto_commit:
            self._auto_commit()

    def _auto_commit(self):
        """Commit the run dir (which now contains the model under model/).

        Lowercase, short message per the project's commit conventions.
        Uses --no-gpg-sign and the configured ottitsch identity.
        """
        run_path = str(Path(self.run_path).relative_to(ROOT))
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "ottitsch",
            "GIT_AUTHOR_EMAIL": "ottitsch.work@gmail.com",
            "GIT_COMMITTER_NAME": "ottitsch",
            "GIT_COMMITTER_EMAIL": "ottitsch.work@gmail.com",
        }
        try:
            subprocess.run(["git", "add", "--", run_path], cwd=ROOT, env=env, check=True)
            msg = f"run {self.run_id.lower()} passed"
            subprocess.run(["git", "commit", "--no-gpg-sign", "-m", msg],
                           cwd=ROOT, env=env, check=True)
            print(f"[auto-commit] committed: {msg}")
        except subprocess.CalledProcessError as e:
            print(f"[auto-commit] skipped: {e}")


if __name__ == "__main__":
    AnimalOutcomeFlow()
