"""
Loaders for the per-year cleaned datasets produced by ``prepare_data.py``.

The training and robustness steps of the flow only ever read data through this
module so that the on-disk layout and feature schema live in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BY_YEAR_DIR = ROOT / "data" / "processed" / "by_year"

# Feature schema. Kept tiny and intentionally simple ("very simple model").
# Categorical columns are one-hot encoded inside the training pipeline; the
# raw column list is what the model contract advertises to consumers.
CATEGORICAL_FEATURES = ["Animal Type", "Sex upon Outcome", "Breed", "Color"]
NUMERIC_FEATURES = ["age_days"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Outcome Type"


def _age_to_days(value: object) -> float | None:
    """Parse strings like '2 years', '3 months', '14 days' into a day count."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parts = str(value).strip().split()
    if len(parts) != 2:
        return None
    try:
        n = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower().rstrip("s")
    factor = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit)
    if factor is None:
        return None
    return n * factor


def load_year(year: int) -> pd.DataFrame:
    """Load a single per-year parquet file with engineered features."""
    path = BY_YEAR_DIR / f"{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset for year {year}: {path}")
    df = pd.read_parquet(path)
    df["age_days"] = df["Age upon Outcome"].apply(_age_to_days)
    return df


def load_years(years: Iterable[int]) -> pd.DataFrame:
    """Concatenate multiple years into one frame."""
    frames = [load_year(y) for y in years]
    return pd.concat(frames, ignore_index=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows missing required features/target and return (X, y)."""
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    clean = df.dropna(subset=needed)
    return clean[FEATURE_COLUMNS].copy(), clean[TARGET_COLUMN].copy()
