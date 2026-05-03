# predict-animal-outcomes

Predict the outcome of an Austin Animal Center intake (Adoption / Transfer /
Return to Owner / ...).

## Quickstart

Requires Python 3.11+ (developed against 3.11.15).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Materialise the parquet datasets (raw + per-year clean splits).
python prepare_data.py

# Run the data quality tests on the raw dataset.
pytest tests/

# Run the full flow: data tests -> train -> robustness validation.
python flow.py run

# Demonstrate the training error path: training raises before fitting and
# the flow exits non-zero. Recovery is just a plain re-run.
python flow.py run --simulate_interrupt True
```

## Repository layout

```
prepare_data.py            CSV -> parquet (raw + per-full-year clean splits)
flow.py                    Metaflow flow: data_tests -> train -> robustness
src/
  data.py                  per-year loaders + feature schema
  training.py              LogisticRegression fit, skops save, register
  registry.py              tiny file-based model registry (list/load/schema)
  robustness.py            holdout-year + majority-baseline validation
tests/                     data quality tests (pytest + great_expectations)
requirements/              per-step requirements.txt for isolated execution
data/
  dataset.parquet          full raw dataset (used by tests/)
  by_year/<year>.parquet   cleaned per-year splits (2014-2024) for training/eval
models/                    registered model versions (v1/, v2/, ...)
```

## Design notes

### Datasets (per-year clean splits)
`prepare_data.py` produces a separate parquet per *full* calendar year
(2014-2024). 2013 (Oct-onward) and 2025 (Jan-May) are partial years and
excluded so per-year evaluations are apples-to-apples. The per-year files have
``Outcome Type`` nulls dropped (the prediction target must be present) and
``Outcome Subtype`` removed (it leaks the label).

### Model and target
- **Target**: `Outcome Type` (multiclass: Adoption, Transfer, Return to Owner, ...).
- **Features**: `Animal Type`, `Sex upon Outcome`, `Breed`, `Color`, `age_days`
  (parsed from `Age upon Outcome`).
- **Model**: multinomial `LogisticRegression` over a one-hot-encoded feature
  matrix. Deliberately tiny -- we just need an artifact to version, load, and
  validate.

### Serialization
The trained `Pipeline(preprocessor + classifier)` is serialized with **skops**,
which is purpose-built for sklearn objects and avoids pickle's
arbitrary-code-execution risk on load.

### Versioning (custom file-based registry)
The task explicitly permits "a very simple own model versioning library" as an
alternative to MLflow. `src/registry.py` is that. Each registered model lives
at `models/v<N>/` with two files:

- `model.skops` -- the serialized sklearn pipeline.
- `schema.json` -- input/output contract (column lists, target column,
  classes), code dependencies (Python version, requirements file), and the
  recorded train accuracy.

The registry exposes the three required operations:
`list_models()`, `load(version)`, `schema(version)`, plus `latest_version()`
for the robustness step.

### Robustness expectation
`src/robustness.py` evaluates the freshly trained model on a holdout year
(default 2024, never seen during training) against two thresholds:

1. **Beat the majority-class baseline by >= 5 percentage points.**
   A `DummyClassifier(strategy="most_frequent")` is the floor below which the
   model adds no value. 5pp is small enough to survive year-to-year class-mix
   noise, large enough that a model that merely re-discovers the prior fails.

2. **Train/holdout accuracy gap <= 10 percentage points.**
   Larger gaps signal overfitting or distribution drift the model hasn't
   generalised across.

Both thresholds are documented inline in `src/robustness.py`. If the model
fails either check, that should prompt a real investigation, not a threshold
tweak.

### Error handling at the training step
Two failure modes are explicitly handled:

- **Too-small training set (< 1000 rows)** raises `ValueError` *before* any
  artifact is written. Refusing to produce a junk model is strictly better
  than logging a warning and registering it: a registered model can be
  promoted to production by accident, a missing model cannot.

- **Simulated mid-training error** (`--simulate_interrupt True`) raises
  `RuntimeError` after data load but before fitting. The flow step propagates
  the error and the model is never registered. Recovery is a plain re-run.
  We don't checkpoint/resume because a one-shot `LogisticRegression.fit` has
  nothing meaningful to checkpoint -- the durable choice is "fail loudly,
  re-run cleanly" rather than half-trained-model gymnastics.

## Per-step dependencies

Each flow step has its own pinned requirements file under `requirements/` so
the steps can be executed in isolation (e.g. inside containers):
- `requirements/data_tests.txt`
- `requirements/train.txt`
- `requirements/robustness.txt`

The top-level `requirements.txt` is the union, used by the local
`python flow.py run` invocation.
