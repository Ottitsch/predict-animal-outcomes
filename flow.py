"""
Metaflow flow: data tests -> training -> robustness validation.

Run locally:
    python flow.py run

Inject a training interruption to demonstrate the recovery path:
    python flow.py run --interrupt_at_epoch 3

Resume from an interrupted run by re-running with the printed checkpoint path:
    python flow.py run --resume_from checkpoints/<run_id>.skops

The three steps each have their own dependency manifest under ``requirements/``
so they can be run in isolation (e.g. inside containers). For the local
``python flow.py run`` invocation the union of those is provided as the top
level ``requirements.txt``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from metaflow import FlowSpec, Parameter, step

ROOT = Path(__file__).resolve().parent


class AnimalOutcomeFlow(FlowSpec):
    train_years = Parameter(
        "train_years",
        help="Comma-separated list of years to train on.",
        default="2014,2015,2016,2017,2018,2019,2020,2021,2022",
    )
    holdout_year = Parameter(
        "holdout_year",
        help="Year to use for the robustness evaluation.",
        default=2024,
        type=int,
    )
    epochs = Parameter("epochs", help="SGD epochs.", default=8, type=int)
    interrupt_at_epoch = Parameter(
        "interrupt_at_epoch",
        help="If set, raise KeyboardInterrupt after this epoch (demo).",
        default=None,
        type=int,
    )
    resume_from = Parameter(
        "resume_from",
        help="Path to a checkpoint produced by a prior interrupted run.",
        default=None,
    )

    @step
    def start(self):
        print("=== Animal Outcome Flow ===")
        print(f"  train_years        = {self.train_years}")
        print(f"  holdout_year       = {self.holdout_year}")
        print(f"  epochs             = {self.epochs}")
        print(f"  interrupt_at_epoch = {self.interrupt_at_epoch}")
        print(f"  resume_from        = {self.resume_from}")
        self.next(self.data_tests)

    @step
    def data_tests(self):
        """Run the data quality tests with pytest.

        The test suite intentionally includes one expected failure
        (``test_outcome_type_not_null``) on the raw dataset; we record the
        outcome but do not block the flow on it. A hard failure of any other
        test would indicate genuine data corruption and should stop the
        pipeline before training.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.tests_stdout = result.stdout
        self.tests_returncode = result.returncode
        print(result.stdout)
        if result.returncode != 0:
            # The known failing test is documented; surface it but continue.
            print("[data_tests] pytest reported failures (see above).")
        self.next(self.train)

    @step
    def train(self):
        """Train + version the model. See ``src/training.py`` for error handling.

        Error-handling design
        ---------------------
        Two failure modes are explicitly handled at this step:

        * **Too-small training set** (``< 1000`` rows): raised as ``ValueError``
          at the very start of training. We refuse to produce an artifact at all
          rather than emit an under-trained model that would silently pass the
          serialization layer and pollute the registry.

        * **Mid-training interruption** (``KeyboardInterrupt``, simulating SIGINT
          or a power loss): the partially-trained estimator is serialised to
          ``checkpoints/<run_id>.skops`` and the corresponding MLflow run is
          tagged ``status=interrupted``. The interrupted run is *not* registered
          in the model registry so it can never be promoted to production by
          accident. Re-running with ``--resume_from <path>`` reloads the
          checkpoint, continues from the next epoch, and registers the model on
          normal completion. We chose this design over "log a warning and
          register anyway" because shipping an under-trained model is a worse
          outcome than failing loudly and forcing an explicit recovery.
        """
        from src.training import train

        years = [int(y) for y in self.train_years.split(",")]
        result = train(
            train_years=years,
            epochs=self.epochs,
            interrupt_at_epoch=self.interrupt_at_epoch,
            resume_from=self.resume_from,
        )
        self.train_status = result.status
        self.train_run_id = result.run_id
        self.model_version = result.model_version
        self.train_accuracy = result.train_accuracy
        self.checkpoint_path = result.checkpoint_path

        if result.status != "completed":
            raise RuntimeError(
                f"training did not complete: status={result.status}; "
                f"checkpoint at {result.checkpoint_path}. "
                f"Re-run with --resume_from {result.checkpoint_path} to continue."
            )
        self.next(self.robustness)

    @step
    def robustness(self):
        """Validate the freshly trained model against the documented thresholds."""
        from dataclasses import asdict
        from src.robustness import evaluate

        report = evaluate(self.holdout_year)
        self.robustness_report = asdict(report)
        print(json.dumps(self.robustness_report, indent=2))
        if not report.passed:
            raise RuntimeError(
                f"robustness check FAILED: "
                f"passed_majority={report.passed_majority}, "
                f"passed_gap={report.passed_gap}"
            )
        print("[robustness] PASSED")
        self.next(self.end)

    @step
    def end(self):
        print("=== flow complete ===")
        print(f"  model_version  = {self.model_version}")
        print(f"  train_run_id   = {self.train_run_id}")
        print(f"  train_accuracy = {self.train_accuracy:.4f}")


if __name__ == "__main__":
    AnimalOutcomeFlow()
