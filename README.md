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
  runs.py                  per-run dirs + flow.json envelope
docker/                    Dockerfiles + build script for host + step containers
requirements/              per-step requirements.txt (one file per container image)
runs/                      per-flow-run audit trail (auto-committed)
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
`src/registry.py` is a small, file-based model registry used in place of
MLflow. Each registered model lives at `models/v<N>/`:

- `model.skops` -- the serialized sklearn pipeline.
- `schema.json` -- input/output contract (column lists, target column,
  classes), code dependencies (Python version, requirements file), the
  recorded train accuracy, the `git_sha` of the commit it was trained from,
  and a `created_at` timestamp.
- `run.json` -- robustness report attached after evaluation
  (holdout accuracy, baseline accuracy, pass/fail flags).

A flat `models/INDEX.json` is regenerated on every write so the full set of
registered versions and their key metadata is reviewable from a single
diff-friendly file.

The registry exposes:
`list_models()`, `load(version)`, `schema(version)`, `attach_run(version, run)`,
plus `latest_version()` for the robustness step.

### Raw data integrity
`raw_data/SHA256SUMS` records the digest of the source CSV. `prepare_data.py`
checks it on every run and refuses to proceed on a mismatch, so a silent
dataset swap is loud rather than silently re-poisoning every downstream
artifact. Regenerate the file only when the source is intentionally updated.

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
the steps execute in isolated containers:
- `requirements/host.txt` (just metaflow, for the driver container)
- `requirements/data_tests.txt`
- `requirements/train.txt`
- `requirements/robustness.txt`

The top-level `requirements.txt` is the union, used only when running the flow
host-natively (`USE_CONTAINERS=0 python flow.py run`).

## Containerized execution

The flow runs as a single host container that spawns a sibling container per
step. Each step image installs only its own `requirements/<step>.txt`.

```
host machine
  /var/run/docker.sock ──┐
  $PWD (the repo)  ──┐   │
                     │   │
       pao-host:dev (driver, runs flow.py)
            │   spawns via docker.sock
            ├──► pao-data-tests:dev   (pytest + great_expectations)
            ├──► pao-train:dev        (sklearn + skops)
            └──► pao-robustness:dev   (sklearn + skops)

  artifacts persist on the host repo via bind mount:
    data/by_year/*.parquet, models/v<N>/*, runs/<run-id>/*
```

### Build + run

```bash
docker/build.sh                                   # ~3 min first time

docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    -e HOST_PROJECT_DIR="$PWD" \
    pao-host:dev run
```

`HOST_PROJECT_DIR` is the absolute path to the repo *on the host machine*,
needed because the bind-mount paths are interpreted by the host's Docker
daemon (not by anything inside the host container).

### How the host knows when a step finishes

`docker run` (foreground) is blocking. The host spawns each step with
`subprocess.Popen` and reads its stdout until the docker CLI exits. Exit code
0 = step passed; non-zero = step failed and the flow stops. No polling, no
healthchecks.

### Per-run audit trail (`runs/`)

Each invocation creates a timestamped, git-sha-tagged directory:

```
runs/<ISO-timestamp>__<short-sha>/
  flow.json                   run-level envelope (parameters, status, timings)
  data_tests/
    stdout.log
    junit.xml
    returncode.txt
  train/
    stdout.log
    summary.json              {"version": N, "train_accuracy": ...}
    schema.json               copy of the registered model schema
    model_version.txt         pointer into models/v<N>/
  robustness/
    stdout.log
    report.json               full RobustnessReport
```

`flow.json` is written incrementally after each step, so a crashed run still
leaves a partial record on disk. On a successful run, `runs/<id>/` and any
new `models/v<N>/` are auto-committed (lowercase, short message, no GPG
signature) — disable with `--auto_commit False`.

### Host-native fallback

```bash
USE_CONTAINERS=0 python flow.py run
```
Skips Docker entirely; each step runs as a plain `subprocess.Popen` on the
host, still capturing stdout into `runs/<run-id>/<step>/stdout.log`.
