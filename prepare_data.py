"""
Serialize the raw CSV dataset to Parquet format in a deterministic location (data/dataset.parquet).
Parquet is chosen for its columnar storage efficiency and native support for typed schemas,
which preserves column types across reads without repeated CSV type inference.
"""
import pandas as pd
from pathlib import Path

RAW_CSV = Path(__file__).parent / "raw_data" / "Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = OUTPUT_DIR / "dataset.parquet"


def prepare():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare()
