"""
Metaflow flow: post-deployment prediction-drift monitoring.

This flow watches an already-registered model rather than producing one. It
loads a model version by run id (default: the latest registered model), scores
an unseen data segment (default: the holdout year 2024), and compares the
predicted-class distribution against the model's training-time reference. See
src/monitoring.py for the full description of what "prediction drift" means
here, where the reference comes from, and why the holdout year is the segment.

Like the training flow, each step shells out to a step-specific container
(``pao-monitor``); the drift report is written under the *monitored model's*
run directory at ``runs/<model-run-id>/monitoring/report.json`` so the signal
stays attached to the exact version it describes.

Run (containers):
    docker/build.sh
    docker run --rm \\
        -v /var/run/docker.sock:/var/run/docker.sock \\
        -v "$PWD":/work -w /work \\
        pao-monitor-host:dev run --model_id <model-run-id>

Run (host-native):
    USE_CONTAINERS=0 python monitor_flow.py run --model_id <model-run-id>
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from metaflow import FlowSpec, Parameter, step

from src import containers


class MonitoringFlow(FlowSpec):
    model_id = Parameter(
        "model_id",
        help="Run id of the model to monitor. Empty -> latest registered model.",
        default="",
    )
    segment_year = Parameter(
        "segment_year",
        help="Unseen data segment to score (default: holdout year 2024).",
        default=2024,
        type=int,
    )

    @step
    def start(self):
        from src import registry as reg
        self.model_run_id = self.model_id or reg.latest_run_id()
        print("=== Monitoring Flow ===")
        print(f"  model_run_id   = {self.model_run_id}")
        print(f"  segment_year   = {self.segment_year}")
        print(f"  use_containers = {containers.USE_CONTAINERS}")
        self.next(self.drift)

    @step
    def drift(self):
        """Score the segment and measure prediction drift in an isolated container."""
        out_subpath = f"runs/{self.model_run_id}/monitoring"
        out_dir = containers.ROOT / out_subpath
        stdout_log = out_dir / "stdout.log"
        cmd = [
            "python", "-m", "src.monitoring",
            "--run-id", self.model_run_id,
            "--segment-year", str(self.segment_year),
            "--out-dir", "/out" if containers.USE_CONTAINERS else str(out_dir),
        ]
        t0 = time.monotonic()
        if containers.USE_CONTAINERS:
            rc = containers.docker_run("monitor", out_subpath, cmd, stdout_log)
        else:
            rc = containers.host_run(cmd, stdout_log)
        duration = round(time.monotonic() - t0, 3)
        report_path = out_dir / "report.json"
        self.report = json.loads(report_path.read_text()) if report_path.exists() else {}
        # rc==1 here means "drift detected", which is a signal, not a flow error;
        # we surface it via the report and a printed warning rather than failing.
        self.drift_detected = rc == 1
        (out_dir / "returncode.txt").write_text(str(rc))
        print(f"[drift] kl={self.report.get('kl_divergence'):.6f} "
              f"threshold={self.report.get('threshold')} "
              f"drift_detected={self.report.get('drift_detected')} "
              f"({duration}s)")
        self.next(self.end)

    @step
    def end(self):
        if self.drift_detected:
            print(f"=== DRIFT DETECTED for {self.model_run_id} "
                  f"(KL={self.report.get('kl_divergence'):.6f}) ===")
        else:
            print(f"=== no drift for {self.model_run_id} ===")


if __name__ == "__main__":
    MonitoringFlow()
