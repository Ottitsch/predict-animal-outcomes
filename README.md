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
monitor_flow.py            Metaflow flow: post-deployment prediction-drift monitor
ab_flow.py                 Metaflow flow: offline A/B comparison of two versions
src/
  data.py                  per-year loaders + feature schema
  training.py              LogisticRegression fit, skops save, register
  registry.py              tiny file-based model registry (list/load/schema)
  robustness.py            holdout-year + majority-baseline validation
  monitoring.py            prediction-drift (KL) against the training reference
  ab_test.py               hash-based traffic split + per-variant scoring
  containers.py            shared sibling-container spawn helpers (monitor/ab)
  runs.py                  per-run dirs + flow.json envelope
tests/                     data quality tests (pytest + great_expectations)
docker/                    Dockerfiles + build script for host + step containers
  requirements/            per-step requirements.txt (one file per container image)
runs/                      per-flow-run audit trail (auto-committed)
  <run-id>/
    data_tests/, train/, robustness/   per-step stdout + reports
    model/                 registered model artifact for this run (skops + schema)
    monitoring/            prediction-drift report for this model (if monitored)
    ab/                    A/B comparison outputs (on A/B-test run dirs)
    flow.json              run envelope
data/
  raw/                     immutable source CSV + SHA256SUMS
  processed/
    dataset.parquet        full raw dataset (used by tests/)
    by_year/<year>.parquet cleaned per-year splits (2014-2024) for training/eval
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

The model + transform choices that affect behaviour (`model_C`,
`model_class_weight`, `model_solver`, `drop_features`, `scale_age`) live in
`config.yml` as flow configuration rather than hard-coded in `training.py`, and
each is overridable as a CLI parameter of the same name. Two runs that differ in
these values are two distinct, independently reproducible model **versions**;
the exact values are recorded in the run envelope and in the model schema
(`schema.json -> hyperparameters`).

### Serialization
The trained `Pipeline(preprocessor + classifier)` is serialized with **skops**,
which is purpose-built for sklearn objects and avoids pickle's
arbitrary-code-execution risk on load.

### Versioning (run-scoped file-based registry)
`src/registry.py` is a small, file-based model registry used in place of
MLflow. Each successful flow run produces exactly one model, persisted next
to that run's audit trail at `runs/<run-id>/model/`:

- `model.skops` -- the serialized sklearn pipeline.
- `schema.json` -- input/output contract (column lists, target column,
  classes), code dependencies (Python version, requirements file), the
  recorded train accuracy, the `git_sha` of the commit it was trained from,
  the originating `run_id`, and a `created_at` timestamp.

Models are identified by their producing `run_id` (timestamped + git-sha
suffixed), not by a separate version namespace. The robustness report for a
run lives at `runs/<run-id>/robustness/report.json` and is *not* duplicated
into the model dir -- everything for a single training run lives under one
parent.

The registry exposes:
`load(run_id)`, `schema(run_id)`, `list_runs_with_model()`,
`latest_run_id()` (used by ad-hoc consumers; the flow always evaluates the
model from its own current run).

### Raw data integrity
`data/raw/SHA256SUMS` records the digest of the source CSV. `prepare_data.py`
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

## Post-deployment monitoring (prediction drift)

`monitor_flow.py` watches an already-registered model instead of producing one.
It loads a model by run id (default: the latest registered model), scores an
unseen data segment, and compares the **predicted-class distribution** against
the model's training-time reference.

```bash
# latest model, default segment (holdout year 2024)
python monitor_flow.py run

# a specific version
python monitor_flow.py run --model_id <run-id>
```

- **What it measures**: prediction (output) drift -- only the distribution of
  classes the model predicts, not features or labels. This is the cheapest
  behaviour-tied signal because it needs only the model and unlabeled input, so
  it runs before ground-truth outcomes are known.
- **Expected ("reference") distribution**: the class mix the model produced on
  its own training data, captured at train time and stored in the schema as
  `train_prediction_distribution`. That is, by construction, the output profile
  the model shipped with -- so anything materially different on fresh data is
  the model behaving differently than at acceptance.
- **Segment**: the holdout year (2024) by default -- the most recent full
  calendar year, never used in training, a clean stand-in for fresh traffic.
- **Metric + threshold**: KL divergence `KL(observed || reference)` in nats,
  flagged at `>= 0.10`. The report (with the full distributions and the
  drift flag) is written to `runs/<model-run-id>/monitoring/report.json`.

Rationale and the threshold choice are documented inline in `src/monitoring.py`.

## A/B testing two versions

`ab_flow.py` compares two registered versions head-to-head on a shared unseen
segment. It forks into two parallel branches (one per version), each scoring
*its half* of the segment, then a join step compares accuracy and macro-F1.

```bash
python ab_flow.py run \
    --run_id_a <version-a-run-id> \
    --run_id_b <version-b-run-id> \
    --test_id my_experiment
```

- **Traffic split**: each row is assigned to a variant by hashing
  `"<test_id>:<Animal ID>"` (MD5) mod 2. This is deterministic/reproducible
  (no RNG, no stored assignment table), uniform (~50/50), and a pure function of
  the id (independent of row order or arrival time).
- **Why salt with `test_id`**: salting the hash gives each experiment its own
  pseudo-random partition, so unit assignments are decorrelated across tests
  rather than the same animals always landing in the same bucket everywhere.
- **Multiple / subsequent tests**: give every experiment a distinct `test_id`.
  That one key both isolates each test's 50/50 partition (as the salt) and tags
  every stored result, so concurrent or later tests never mix -- filter results
  by `test_id` to read a single experiment.
- **Outputs**: `variant_a.json`, `variant_b.json`, and the joined
  `comparison.json` (per-metric deltas + winner by macro-F1) under
  `runs/<ab-run-id>/ab/`.

The split, salting, and multi-test reasoning are documented inline in
`src/ab_test.py`.

## Per-step dependencies

Each flow step has its own pinned requirements file under `docker/requirements/`
so the steps execute in isolated containers:
- `docker/requirements/host.txt` (just metaflow, for the driver containers)
- `docker/requirements/data_tests.txt`
- `docker/requirements/train.txt`
- `docker/requirements/robustness.txt`
- `docker/requirements/monitor.txt` (sklearn + skops, for the drift monitor)
- `docker/requirements/ab.txt` (sklearn + skops, for A/B scoring)

The top-level `requirements.txt` is the union, used only when running the flow
host-natively (`USE_CONTAINERS=0 python flow.py run`).

## Containerized execution

The flow runs as a single host container that spawns a sibling container per
step. Each step image installs only its own `docker/requirements/<step>.txt`.

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
    data/processed/by_year/*.parquet, runs/<run-id>/* (incl. model/)
```

### Build + run

```bash
docker/build.sh                                   # ~3 min first time

docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-host:dev run
```

The monitoring and A/B flows follow the same host-spawns-step pattern, each with
its own driver image (`pao-monitor-host`, `pao-ab-host`) spawning a step image
(`pao-monitor`, `pao-ab`):

```bash
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-monitor-host:dev run --model_id <run-id>

docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-ab-host:dev run --run_id_a <id-a> --run_id_b <id-b> --test_id exp1
```

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
    report.json
    returncode.txt
  train/
    stdout.log
    summary.json              {"run_id": "...", "train_accuracy": ...}
  model/
    model.skops               serialized sklearn pipeline
    schema.json               input/output contract + run_id + git_sha
  robustness/
    stdout.log
    report.json               full RobustnessReport
```

`flow.json` is written incrementally after each step, so a crashed run still
leaves a partial record on disk. The model artifact is colocated with its run
so provenance is one `cd` away in either direction. On a successful run,
`runs/<id>/` is auto-committed (lowercase, short message, no GPG signature)
— disable with `--auto_commit False`.

### Host-native fallback

```bash
USE_CONTAINERS=0 python flow.py run
USE_CONTAINERS=0 python monitor_flow.py run --model_id <run-id>
USE_CONTAINERS=0 python ab_flow.py run --run_id_a <id-a> --run_id_b <id-b>
```
Skips Docker entirely; each step runs as a plain `subprocess.Popen` on the
host, still capturing stdout into the run dir's per-step `stdout.log`.
