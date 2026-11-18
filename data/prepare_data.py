"""
Build the training data: one row per shelter stay, with only what is known at intake.

Why stays and not outcome rows
------------------------------
The first version of this project trained on the Outcomes file alone. Its
columns are recorded when the animal *leaves* the shelter, and two of them gave
the answer away:

* ``Sex upon Outcome`` says whether the animal was neutered by the time it
  left. The shelter neuters animals before adoption, so "Neutered Male" and
  "Spayed Female" rows are 67% and 72% adoptions while intact animals are 8%.
* ``Age upon Outcome`` includes the length of the stay itself.

That model scored 0.81 on 2024, but a model that is asked at intake can't see
either column. So we now join the Intakes file and predict each stay's outcome
from intake time columns only (see ``predict_animal_outcomes/data.py``).

How a stay is built
-------------------
Each intake is paired with the first outcome of the same ``Animal ID`` at or
after it, as long as that outcome comes before the animal's next intake. An
intake with no such outcome (the animal was still in the shelter when the
snapshot was taken, or the record is missing) is dropped, and so is an
outcome with no intake before it.

Timestamps are read as Austin wall clock time in both files. The Outcomes
export mixes rows with a ``-05:00`` offset and rows with none; parsing them
with ``utc=True`` moves the rows without an offset by five to six hours and
made about 10% of outcomes look as if they happened before their intake.

Outputs
-------
``data/processed/by_year/<year>.parquet``  stays by *intake* year, full years
2014 to 2024 only (2013 starts in October and 2025 ends in May).
``data/processed/join_report.json``        how many rows matched, for the README.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# This script lives in data/; the repo root is its parent.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTCOMES_CSV = RAW_DIR / "Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv"
INTAKES_CSV = RAW_DIR / "Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv"
RAW_SHA = RAW_DIR / "SHA256SUMS"
OUTPUT_DIR = ROOT / "data" / "processed"
BY_YEAR_DIR = OUTPUT_DIR / "by_year"
JOIN_REPORT = OUTPUT_DIR / "join_report.json"

# Full calendar years of intakes in the source data.
FULL_YEARS = list(range(2014, 2025))

# Columns kept per stay. Everything except the outcome columns is known at intake.
STAY_COLUMNS = [
    "Animal ID", "intake_time", "Animal Type", "Intake Type", "Intake Condition",
    "Sex upon Intake", "Age upon Intake", "Breed", "Color",
    "outcome_time", "Outcome Type",
]


def _verify_raw_sha() -> None:
    """Fail loudly if a raw CSV is missing from SHA256SUMS or doesn't match it.

    A silent CSV swap would invalidate every downstream artifact (parquet,
    trained model, tests). Checking once at the entry point is cheap.
    """
    expected = {}
    for line in RAW_SHA.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    for csv in (OUTCOMES_CSV, INTAKES_CSV):
        want = expected.get(csv.name)
        if want is None:
            raise RuntimeError(f"{csv.name} has no entry in {RAW_SHA}")
        actual = hashlib.sha256(csv.read_bytes()).hexdigest()
        if actual != want:
            raise RuntimeError(
                f"SHA mismatch for {csv.name}: expected {want}, got {actual}. "
                f"Refusing to proceed; regenerate {RAW_SHA} only if the source "
                f"dataset has been intentionally updated."
            )


def build_stays(intakes: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pair every intake with its outcome. Returns (stays, counts for the join report)."""
    intakes = intakes.drop_duplicates().copy()
    outcomes = outcomes.drop_duplicates().copy()
    intakes["intake_time"] = pd.to_datetime(intakes["DateTime"], format="%m/%d/%Y %I:%M:%S %p")
    # Wall clock as written: drop any "-05:00" style offset, see the module docstring.
    outcomes["outcome_time"] = pd.to_datetime(outcomes["DateTime"].str[:19], format="%Y-%m-%dT%H:%M:%S")
    outcomes = outcomes.dropna(subset=["Outcome Type"])

    intakes = intakes.sort_values(["Animal ID", "intake_time"])
    intakes["next_intake_time"] = intakes.groupby("Animal ID")["intake_time"].shift(-1)

    paired = pd.merge_asof(
        intakes.sort_values("intake_time"),
        outcomes[["Animal ID", "outcome_time", "Outcome Type"]].sort_values("outcome_time"),
        left_on="intake_time", right_on="outcome_time", by="Animal ID",
        direction="forward",
    )
    has_outcome = paired["outcome_time"].notna()
    before_next = paired["next_intake_time"].isna() | (paired["outcome_time"] < paired["next_intake_time"])
    stays = paired[has_outcome & before_next][STAY_COLUMNS].reset_index(drop=True)

    counts = {
        "intakes": int(len(intakes)),
        "outcomes_with_type": int(len(outcomes)),
        "stays": int(len(stays)),
        "intakes_without_outcome": int((~has_outcome).sum()),
        "intakes_whose_next_outcome_follows_a_later_intake": int((has_outcome & ~before_next).sum()),
        "outcomes_not_in_any_stay": int(len(outcomes) - len(stays)),
    }
    return stays, counts


def _write_parquet(df: pd.DataFrame, path: Path, extra_meta: dict | None = None) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"created_at": datetime.now(timezone.utc).isoformat()}
    if extra_meta:
        meta.update(extra_meta)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def prepare() -> None:
    BY_YEAR_DIR.mkdir(parents=True, exist_ok=True)
    _verify_raw_sha()

    stays, counts = build_stays(pd.read_csv(INTAKES_CSV), pd.read_csv(OUTCOMES_CSV))
    print(f"{counts['stays']} stays from {counts['intakes']} intakes "
          f"and {counts['outcomes_with_type']} outcomes")

    year = stays["intake_time"].dt.year
    counts["stays_per_intake_year"] = {}
    for y in FULL_YEARS:
        chunk = stays[year == y]
        out = BY_YEAR_DIR / f"{y}.parquet"
        _write_parquet(chunk, out, extra_meta={"year": str(y), "rows": str(len(chunk))})
        counts["stays_per_intake_year"][str(y)] = int(len(chunk))
        print(f"  {y}: {len(chunk):>6d} stays -> {out}")
    JOIN_REPORT.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    prepare()
