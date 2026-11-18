# predict-animal-outcomes

[![tests](https://github.com/Ottitsch/predict-animal-outcomes/actions/workflows/tests.yml/badge.svg)](https://github.com/Ottitsch/predict-animal-outcomes/actions/workflows/tests.yml)

Predicts how a stay at the Austin Animal Center will end (Adoption, Transfer,
Return to Owner, ...) from what is known when the animal arrives. The model is
deliberately simple; the project is the pipeline around it: Metaflow flows
with one Docker image per step, data tests that gate training, a file based
model registry, a robustness gate on a held out year, prediction drift
monitoring and an offline A/B test with bootstrap confidence intervals.

## Results

Holdout: the 11,659 stays that began in 2024. Model: multinomial logistic
regression trained on 2014 to 2023 stays.

| | accuracy | macro F1 |
|---|---|---|
| majority class (always "Adoption") | 0.624 | 0.085 |
| **logistic regression, intake features** | **0.697** | **0.266** |

Recall per outcome: Adoption 0.87, Euthanasia 0.55, Return to Owner 0.52,
Transfer 0.41. The five rarest outcomes (Died, Rto-Adopt, Disposal, Missing,
Lost: 3.1% of 2024 stays together) are never predicted correctly.

All numbers come from
[`runs/2026-11-18T14-05-43Z__8ac8d1d/robustness/report.json`](runs/2026-11-18T14-05-43Z__8ac8d1d/robustness/report.json).

## v2: the first model leaked the label

The first version trained on the Outcomes file alone and scored 0.81 on 2024,
18 points above the majority baseline. Those columns are recorded when the
animal *leaves*, and one of them gave the answer away: `Sex upon Outcome` says
whether the animal was neutered by then, and the shelter neuters animals
before adoption. Neutered and spayed animals are 67% and 72% adoptions,
intact ones about 8%. `Age upon Outcome` also includes the length of the stay.

| model (train 2014 to 2023, evaluate 2024) | accuracy | margin over majority | macro F1 |
|---|---|---|---|
| v1: outcome time columns | 0.812 | +18.3 points | 0.289 |
| v1 without `Sex upon Outcome` | 0.636 | +0.7 points | 0.198 |
| **v2: intake time columns** | **0.697** | **+7.3 points** | **0.266** |

`python analysis/leakage_check.py` reproduces this table (on Linux; the
solver stops at a tolerance, so the last digit can differ on other machines).
v2 joins the Intakes
snapshot, pairs every intake with its outcome into a stay, and uses only what
is known at intake: animal type, intake type, intake condition, sex upon
intake, age upon intake, breed and color.

Reviewing v1 turned up more problems, all fixed in v2:

* **The A/B test picked winners from noise.** It compared two models on two
  different halves of 2024 and called the higher macro F1 the winner. The
  halves alone differ by a few points by chance. v2 also scores both versions
  on every row and only names a winner when a paired bootstrap interval
  excludes zero ([details](#ab-testing-two-versions)).
* **The train/holdout gap was checked in one direction only**, so a holdout
  10.6 points *above* train passed. The adoption share rose from 38% of 2014
  stays to 62% of 2024 stays, and the gate couldn't see that.
* **The data tests could never block a run**, and their adoption rate bound
  pooled all years, hiding that 2024 was above it. They now run per
  year and a failure stops the flow.
* **The raw data checksum failed on every Windows clone**: git rewrote the CSV
  line endings on checkout. `.gitattributes` now marks raw files as binary.
* **The drift monitor used one exit code for "drift" and "crashed"**, and
  could read a stale report after a crash. Drift is now exit 2.
* **Models were loaded with every type the file declared trusted**, which
  undid the point of skops. They now load with skops' default trust.
* Workflow inputs were pasted into shell commands; they now go through env.

The v1 runs are in the git history before commit `ffdb571`.

## Quickstart

Requires Python 3.11+ (the Docker images use 3.13). Metaflow imports `fcntl`,
so the flows run on Linux and macOS; on Windows use WSL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the per year stay files from the raw CSVs (checks their SHA256 first).
python data/prepare_data.py

# Unit tests (stay join, A/B statistics) and data tests (the stay files).
python -m pytest tests/unit
python -m pytest tests/ --ignore=tests/unit

# Full flow: data tests -> train -> robustness, without Docker.
USE_CONTAINERS=0 python flows/flow.py run

# Demonstrate the training error path: training raises before fitting and
# the flow exits non-zero. Recovery is just a plain re-run.
USE_CONTAINERS=0 python flows/flow.py run --simulate_interrupt True

# Reproduce the v1 versus v2 table above.
python analysis/leakage_check.py
```

## Repository layout

```
flows/                     Metaflow flow drivers (entry points)
  flow.py                  data_tests -> train -> robustness
  monitor_flow.py          post-deployment prediction-drift monitor
  ab_flow.py               offline A/B comparison of two versions
  training_defaults.yml    default training-flow config (overridable per run)
predict_animal_outcomes/
  data.py                  per-year loaders + feature schema
  training.py              LogisticRegression fit, skops save, register
  registry.py              tiny file-based model registry (list/load/schema)
  robustness.py            holdout-year + majority-baseline validation
  monitoring.py            prediction-drift (KL) against the training reference
  ab_test.py               hash-based traffic split, scoring, bootstrap comparison
  containers.py            shared sibling-container spawn helpers (all flows)
  runs.py                  per-run dirs + flow.json envelope
analysis/
  leakage_check.py         reproduces the v1 versus v2 table
tests/                     data quality tests on the stay files (pytest + great_expectations)
  unit/                    unit tests for the stay join and the A/B statistics
docker/                    Dockerfiles + build script for host + step containers
  requirements/            per-step requirements.txt (one file per container image)
runs/                      per-flow-run audit trail (committed)
  <run-id>/
    data_tests/, train/, robustness/   per-step stdout + reports
    model/                 registered model artifact for this run (skops + schema)
    monitoring/            prediction-drift report for this model (if monitored)
    ab/                    A/B comparison outputs (on A/B-test run dirs)
    flow.json              run envelope
data/
  prepare_data.py          raw CSVs -> per year stay files
  raw/                     immutable source CSVs + SHA256SUMS
  processed/
    by_year/<year>.parquet stays by intake year (2014-2024) for training/eval
    join_report.json       how many intakes and outcomes became stays
```

## Design notes

### Data: one row per stay
`data/prepare_data.py` pairs every intake with the first outcome of the same
`Animal ID` at or after it, as long as that outcome comes before the animal's
next intake. 172,098 of 173,776 intakes become stays; 1,622 intakes have no
outcome yet (the animal was still in the shelter on 2025-05-05, or the record
is missing) and are dropped, as are outcomes with no intake before them
(`data/processed/join_report.json` has the counts).

Timestamps are read as Austin wall clock time. The Outcomes export mixes rows
with a `-05:00` offset and rows with none; parsing them as UTC moved the
naive rows by five to six hours and made about 10% of outcomes appear before
their own intake. A data test now fails if any stay ends before it starts.

Stays are split by **intake** year, the moment the prediction would be made.
Only full calendar years are kept (2014 to 2024): 2013 starts in October and
2025 ends in May. A side effect worth knowing: 2024 intakes still open on
2025-05-05 are missing from the holdout, so it slightly favours shorter stays.

### Model and target
- **Target**: `Outcome Type` (multiclass: Adoption, Transfer, Return to Owner, ...).
- **Features**: `Animal Type`, `Intake Type`, `Intake Condition`,
  `Sex upon Intake`, `Breed`, `Color`, `age_days` (parsed from
  `Age upon Intake`). Intake month and the number of earlier intakes were
  tried and changed holdout accuracy by less than 0.1 points.
- **Model**: multinomial `LogisticRegression` over a one-hot-encoded feature
  matrix. Deliberately simple: the project is about the pipeline around it.

The model + transform choices that affect behaviour (`model_C`,
`model_class_weight`, `model_solver`, `drop_features`, `scale_age`) live in
`flows/training_defaults.yml` as flow configuration rather than hard-coded in `training.py`, and
each is overridable as a CLI parameter of the same name. Two runs that differ in
these values are two distinct, independently reproducible model **versions**;
the exact values are recorded in the run envelope and in the model schema
(`schema.json -> hyperparameters`).

### Serialization
The trained `Pipeline(preprocessor + classifier)` is serialized with **skops**,
which is purpose-built for sklearn objects and avoids pickle's
arbitrary-code-execution risk on load. `registry.load` passes no extra trusted
types, so a file that needs anything outside skops' defaults is refused.

### Versioning (run-scoped file-based registry)
`predict_animal_outcomes/registry.py` is a small, file-based model registry used in place of
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
`data/raw/SHA256SUMS` records the digest of both source CSVs. `prepare_data.py`
checks them on every run and refuses to proceed on a mismatch or a missing
entry, so a silent dataset swap is loud rather than silently re-poisoning every
downstream artifact. `.gitattributes` keeps git from rewriting their line
endings, which would change the digest.

### Data tests
`tests/` checks the exact stay files a run trains and evaluates on (its train
years plus its holdout year): required columns never null, descriptive ones at
least 95% present, every value of Animal Type, Intake Type and Outcome Type in
its known set, dogs and cats at least 90% of each year, the adoption share of
each year between 30% and 75%, and no stay ending before it starts. The
rationale for each bound is in the test docstrings. A failing test stops the
flow before training.

### Robustness expectation
`predict_animal_outcomes/robustness.py` evaluates the freshly trained model on a holdout year
(default 2024, never seen during training) against two thresholds:

1. **Beat the majority-class baseline by >= 5 percentage points.**
   A `DummyClassifier(strategy="most_frequent")` fitted on the holdout year
   is the floor below which the model adds no value. 5pp is small enough to
   survive year-to-year class-mix noise, large enough that a model that merely
   re-discovers the prior fails.

2. **Train/holdout accuracy gap within 10 percentage points, either way.**
   Train far above holdout means overfitting to the training years. Holdout
   far above train means the holdout year looks different from the training
   years, so accuracy on it says little about the next one.

The report also carries macro F1 for the model and the baseline, the holdout
class mix and the recall of every class. Both thresholds are documented
inline in `predict_animal_outcomes/robustness.py`. If the model fails either
check, that should prompt a real investigation, not a threshold tweak.

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
python flows/monitor_flow.py run

# a specific version
python flows/monitor_flow.py run --model_id <run-id>
```

- **What it measures**: prediction (output) drift -- only the distribution of
  classes the model predicts, not features or labels. This is the cheapest
  behaviour-tied signal because it needs only the model and unlabeled input, so
  it runs before ground-truth outcomes are known.
- **Expected ("reference") distribution**: the class mix the model produced on
  its own training data, captured at train time and stored in the schema as
  `train_prediction_distribution`.
- **Segment**: the holdout year (2024) by default -- the most recent full
  calendar year, never used in training. The data here is a frozen snapshot,
  so the weekly scheduled run rescores the same year and demonstrates the
  mechanism; in production the segment would be the latest week of intakes.
- **Metric + threshold**: KL divergence `KL(observed || reference)` in nats,
  flagged at `>= 0.10`. The report (with the full distributions and the
  drift flag) is written to `runs/<model-run-id>/monitoring/report.json`.
  The current model sits at 0.019.
- **Exit codes**: 0 no drift, 2 drift (reported, not a flow failure), anything
  else is a crash and fails the flow.

Rationale and the threshold choice are documented inline in `predict_animal_outcomes/monitoring.py`.

## A/B testing two versions

`ab_flow.py` compares two registered versions on a shared unseen segment. It
forks into two parallel branches (one per version), each scoring the segment
in the ab container, then a join step compares them, also in the ab container.

```bash
python flows/ab_flow.py run \
    --run_id_a <version-a-run-id> \
    --run_id_b <version-b-run-id> \
    --test_id my_experiment
```

- **Traffic split**: each row is assigned to a variant by hashing
  `"<test_id>:<Animal ID>"` (MD5) mod 2. This is deterministic/reproducible
  (no RNG, no stored assignment table), uniform (~50/50), and a pure function of
  the id (independent of row order or arrival time). Salting with `test_id`
  gives every experiment its own partition and tags its results.
- **Two views of the result**: the *split* view (A on its half, B on the
  other) is what an online test would see. Offline both versions can score
  every row, so the *paired* view compares them on identical rows. Both get a
  bootstrap 95% interval (2,000 resamples, fixed seed).
- **Decision**: a winner is only named when the paired interval for the
  macro F1 difference excludes zero; otherwise "no significant difference".

The committed test compares training on all years (A) with training on 2019
to 2023 only (B) ([`comparison.json`](runs/2026-11-18T14-08-20Z__8ac8d1d/ab/comparison.json)):

| view | macro F1, B minus A | 95% interval |
|---|---|---|
| split (5,870 vs 5,789 rows) | -0.029 | -0.043 to +0.020 |
| paired (11,659 rows, 576 predictions differ) | +0.002 | -0.002 to +0.006 |

On the split view alone A looks 0.029 better, the kind of gap v1 reported as
a win; on identical rows the two are indistinguishable. Decision: no significant difference.

Outputs under `runs/<ab-run-id>/ab/`: `variant_a.json`, `variant_b.json`,
`predictions_a.csv`, `predictions_b.csv` and `comparison.json`.

## Per-step dependencies

Each flow step has its own pinned requirements file under `docker/requirements/`
so the steps execute in isolated containers:
- `docker/requirements/host.txt` (just metaflow, for the driver containers)
- `docker/requirements/data_tests.txt`
- `docker/requirements/train.txt`
- `docker/requirements/robustness.txt`
- `docker/requirements/monitor.txt` (sklearn + skops, for the drift monitor)
- `docker/requirements/ab.txt` (sklearn + skops, for A/B scoring)

The top-level `requirements.txt` is the union, used when running the flows
host-natively (`USE_CONTAINERS=0`) and by the tests workflow.

## Containerized execution

Each flow runs as a single host (driver) container that spawns a sibling
container per step. There is **one** host image (`pao-host`): it carries only
metaflow + the docker CLI and launches whichever flow you pass as its command,
since the driver is an orchestrator, not a pipeline step. Each *step* image
installs only its own `docker/requirements/<step>.txt`.

```
host machine
  /var/run/docker.sock ──┐
  $PWD (the repo)  ──┐   │
                     │   │
       pao-host:dev  (driver; runs flows/flow.py | monitor_flow.py | ab_flow.py)
            │   spawns via docker.sock
            ├──► pao-data-tests:dev   (pytest + great_expectations)
            ├──► pao-train:dev        (sklearn + skops)
            ├──► pao-robustness:dev   (sklearn + skops)
            ├──► pao-monitor:dev      (sklearn + skops)   [monitor_flow]
            └──► pao-ab:dev           (sklearn + skops)   [ab_flow]

  artifacts persist on the host repo via bind mount:
    data/processed/by_year/*.parquet, runs/<run-id>/* (incl. model/)
```

### Build + run

`docker/build.sh [tag] [name ...]` builds every image by default, or only the
named ones so a workflow builds just what it runs:

```bash
docker/build.sh                                   # all images (~3 min first time)
docker/build.sh dev host data-tests train robustness   # just the training pipeline

# training flow
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-host:dev flows/flow.py run

# monitoring flow (one host image, flow passed as the command)
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-host:dev flows/monitor_flow.py run --model_id <run-id>

# A/B flow
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD":/work -w /work \
    pao-host:dev flows/ab_flow.py run --run_id_a <id-a> --run_id_b <id-b> --test_id exp1
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

The committed runs were produced host-natively in WSL (`use_containers: false`
in their `flow.json`); the GitHub Actions workflows below produce container runs.

### Host-native fallback

```bash
USE_CONTAINERS=0 python flows/flow.py run
USE_CONTAINERS=0 python flows/monitor_flow.py run --model_id <run-id>
USE_CONTAINERS=0 python flows/ab_flow.py run --run_id_a <id-a> --run_id_b <id-b>
```
Skips Docker entirely; each step runs as a plain `subprocess.Popen` on the
host, still capturing stdout into the run dir's per-step `stdout.log`.

## GitHub Actions

- **`.github/workflows/tests.yml`** (`tests`, every push and pull request):
  unit tests, data tests, and a rebuild of the stay files from the raw CSVs
  that must reproduce the committed join counts.
- **`.github/workflows/pipeline.yml`** (`training-pipeline`, manual): builds the
  training images and runs the training flow (data tests -> train ->
  robustness). This is the flow that *produces* model versions.
- **`.github/workflows/monitoring.yml`** (`monitoring`, scheduled + manual):
  builds the monitor image and runs the prediction-drift monitor. Takes an
  optional `model_id` (which model to monitor; empty -> latest) as a dispatch
  input.
- **`.github/workflows/ab-test.yml`** (`ab-test`, manual): builds the ab image
  and runs the offline A/B comparison of two versions. Takes `run_id_a` /
  `run_id_b` (the two versions to compare) and `test_id` as dispatch inputs.

The pipeline, monitoring and A/B workflows commit their `runs/` artifacts back
to the branch they ran on. Dispatch inputs reach the shell only through
environment variables.

## License

MIT, see [LICENSE](LICENSE).
