"""
Prediction-drift monitor for a registered model.

What this measures
------------------
This is a **prediction-drift** check (a.k.a. output drift). It does *not* look
at the input features or the true labels; it looks only at the *distribution of
the classes the model predicts*. Concretely:

  * **Reference ("expected") distribution** -- the class mix the model produced
    when it scored its own training data. It is computed at training time and
    stored in the model schema as ``train_prediction_distribution`` (see
    predict_animal_outcomes/training.py). We source the expectation from the model's own training
    behaviour because that is, by construction, the output profile the model was
    shipped with: whatever mix of Adoption / Transfer / Return-to-Owner / ...
    it emitted on the data it learned from is the behaviour we implicitly
    accepted when we registered it. Anything materially different on fresh data
    is, by definition, the model behaving differently than it did at acceptance.

  * **Observed distribution** -- the class mix the same model produces on a
    segment of data it never saw during training (default: the holdout year,
    2024). See ``DRIFT_SEGMENT_NOTE`` for why that segment is the right one.

We summarise the gap with the **Kullback-Leibler divergence** ``KL(observed ||
reference)`` over the class set, in nats. KL is asymmetric and we deliberately
put the *observed* distribution first: it penalises the model for putting
probability mass on outcomes that were rare at training time, which is the
direction we care about (a sudden surge of, say, Euthanasia predictions). Both
distributions are Laplace-smoothed with a small epsilon so a class that is
absent on one side cannot send the divergence to infinity.

Why prediction drift (rather than feature or label drift)?
  Prediction drift is the cheapest signal that is still directly tied to model
  behaviour: it needs only the model and unlabeled input, so it can run in
  production before ground-truth outcomes are known. A jump here is an early
  warning that *something* upstream changed enough to move the model's outputs,
  and is the natural trigger for a deeper (feature- or label-level) look.

Expectation / threshold
------------------------
We flag drift when ``KL >= DRIFT_THRESHOLD`` (0.10 nats). On stable data the two
distributions are nearly identical and KL sits far below this; 0.10 nats is
loose enough to ignore ordinary year-to-year wobble but trips on a real shift in
the output mix. As with the robustness thresholds, a trip should prompt an
investigation, not a threshold tweak.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from . import data as data_mod
from . import registry as reg

DRIFT_THRESHOLD = 0.10  # nats
_EPS = 1e-6

DRIFT_SEGMENT_NOTE = (
    "Segment = the holdout year (2024). It is the most recent full calendar "
    "year and was never part of any training run (training uses 2014-2023), so "
    "the model's predictions on it are a clean stand-in for 'fresh production "
    "traffic'. Partial years (2013, 2025) are excluded upstream so the segment "
    "is a like-for-like full year, not a seasonal slice."
)


@dataclass
class DriftReport:
    run_id: str
    segment_year: int
    n_segment_rows: int
    reference_distribution: dict
    observed_distribution: dict
    kl_divergence: float
    threshold: float
    drift_detected: bool
    segment_note: str


def _kl(observed: dict, reference: dict, classes: list[str]) -> float:
    """KL(observed || reference) in nats, with epsilon smoothing on both sides."""
    p = {c: observed.get(c, 0.0) + _EPS for c in classes}
    q = {c: reference.get(c, 0.0) + _EPS for c in classes}
    p_tot, q_tot = sum(p.values()), sum(q.values())
    return sum((p[c] / p_tot) * math.log((p[c] / p_tot) / (q[c] / q_tot))
               for c in classes)


def evaluate(run_id: str, segment_year: int) -> DriftReport:
    model = reg.load(run_id)
    schema = reg.schema(run_id)
    reference = schema.get("train_prediction_distribution")
    if not reference:
        raise RuntimeError(
            f"model {run_id} has no train_prediction_distribution in its schema; "
            "retrain with the current training code to capture the drift reference"
        )

    df = data_mod.load_year(segment_year)
    X, _ = data_mod.split_xy(df)
    if len(X) == 0:
        raise RuntimeError(f"segment year {segment_year} has no usable rows")

    preds = model.predict(X)
    classes = sorted(set(reference) | {str(p) for p in preds})
    counts = {c: 0 for c in classes}
    for p in preds:
        counts[str(p)] += 1
    total = float(len(preds))
    observed = {c: counts[c] / total for c in classes}

    kl = _kl(observed, reference, classes)
    return DriftReport(
        run_id=run_id,
        segment_year=segment_year,
        n_segment_rows=len(X),
        reference_distribution=reference,
        observed_distribution=observed,
        kl_divergence=kl,
        threshold=DRIFT_THRESHOLD,
        drift_detected=kl >= DRIFT_THRESHOLD,
        segment_note=DRIFT_SEGMENT_NOTE,
    )


def _cli() -> int:
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(
        description="Prediction-drift monitor for a registered model. Invoked "
                    "by monitor_flow.py inside the monitor container.",
    )
    p.add_argument("--run-id", required=True,
                   help="Run id whose model should be monitored.")
    p.add_argument("--segment-year", type=int, default=2024,
                   help="Unseen data segment to score (default: holdout 2024).")
    p.add_argument("--out-dir", default=None,
                   help="If set, write the drift report JSON here.")
    args = p.parse_args()
    report = evaluate(args.run_id, args.segment_year)
    payload = asdict(report)
    print(json.dumps(payload, indent=2))
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(payload, indent=2))
    # Non-zero exit signals drift so the flow can surface it; the report is
    # always written first so the audit trail is complete either way.
    return 1 if report.drift_detected else 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
