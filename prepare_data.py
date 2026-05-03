"""
Serialize the raw CSV dataset into Parquet files.

Two artifacts are produced:

1. ``data/dataset.parquet`` -- the full raw dataset, used by the data quality
   tests under ``tests/`` (which intentionally run against the *uncleaned* data).

2. ``data/by_year/<year>.parquet`` -- a cleaned, per-full-year split used for
   model training and evaluation. Cleaning consists of:
     - Dropping rows where ``Outcome Type`` is null (target is required).
     - Dropping the ``Outcome Subtype`` column (would leak the label).
   Only *complete* calendar years are emitted (2014-2024); 2013 and 2025 are
   partial in the source data and are excluded so per-year comparisons are
   apples-to-apples.

Parquet is chosen for columnar efficiency and for preserving column types so
that the type of ``DateTime`` does not have to be re-inferred on every read.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "raw_data" / "Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv"
OUTPUT_DIR = ROOT / "data"
RAW_PARQUET = OUTPUT_DIR / "dataset.parquet"
BY_YEAR_DIR = OUTPUT_DIR / "by_year"

# Full calendar years available in the source data. 2013 starts in October and
# 2025 ends in May, so neither is a full year and both are excluded.
FULL_YEARS = list(range(2014, 2025))


def _write_parquet(df: pd.DataFrame, path: Path, extra_meta: dict | None = None) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {"created_at": datetime.now(timezone.utc).isoformat()}
    if extra_meta:
        meta.update(extra_meta)
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def prepare() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    BY_YEAR_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    _write_parquet(df, RAW_PARQUET)
    print(f"Saved {len(df)} rows to {RAW_PARQUET}")

    clean = df.dropna(subset=["Outcome Type"]).copy()
    if "Outcome Subtype" in clean.columns:
        clean = clean.drop(columns=["Outcome Subtype"])
    clean["_year"] = pd.to_datetime(clean["DateTime"], format="ISO8601", utc=True).dt.year

    for year in FULL_YEARS:
        chunk = clean[clean["_year"] == year].drop(columns=["_year"])
        out = BY_YEAR_DIR / f"{year}.parquet"
        _write_parquet(chunk, out, extra_meta={"year": str(year), "rows": str(len(chunk))})
        print(f"  {year}: {len(chunk):>6d} rows -> {out}")


if __name__ == "__main__":
    prepare()
