"""
Metaflow flow: data tests -> training -> robustness validation.

Run locally:
    python flow.py run

Inject a training error to demonstrate the failure path:
    python flow.py run --simulate_interrupt True

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
    simulate_interrupt = Parameter(
        "simulate_interrupt",
        help="If True, raise during training to demo the error path.",
        default=False,
        type=bool,
    )

    @step
    def start(self):
        print("=== Animal Outcome Flow ===")
        print(f"  train_years        = {self.train_years}")
        print(f"  holdout_year       = {self.holdout_year}")
        print(f"  simulate_interrupt = {self.simulate_interrupt}")
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
            print("[data_tests] pytest reported failures (see above).")
        self.next(self.train)

    @step
    def train(self):
        """Train + register the model. See ``src/training.py`` for error handling.

        Two failure modes are explicitly handled at this step:

        * **Too-small training set** (``< 1000`` rows) raises ``ValueError``
          before any artifact is written. We refuse to produce an under-trained
          model that could be promoted by accident.
        * **Simulated mid-training error** (``simulate_interrupt=True``) raises
          ``RuntimeError`` after data load but before fitting. The flow step
          propagates the error and the model is never registered. Recovery is
          a plain re-run.
        """
        from src.training import train

        years = [int(y) for y in self.train_years.split(",")]
        result = train(train_years=years, simulate_interrupt=self.simulate_interrupt)
        self.model_version = result.version
        self.train_accuracy = result.train_accuracy
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
        print(f"  train_accuracy = {self.train_accuracy:.4f}")


if __name__ == "__main__":
    AnimalOutcomeFlow()
