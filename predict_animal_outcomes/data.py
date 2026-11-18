"""
Loaders for the per-year stay datasets produced by ``data/prepare_data.py``.

The training and robustness steps of the flow only ever read data through this
module so that the on-disk layout and feature schema live in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BY_YEAR_DIR = ROOT / "data" / "processed" / "by_year"

# Feature schema. Every feature is known when the animal arrives, because that
# is when the prediction would be made. The first version used "Sex upon
# Outcome" and "Age upon Outcome" instead, which are recorded when the animal
# leaves and leak the answer (see data/prepare_data.py). Intake month and the
# number of earlier intakes were tried and moved holdout accuracy by less than
# 0.1 points, so they are left out.
# Categorical columns are one-hot encoded inside the training pipeline; the
# raw column list is what the model contract advertises to consumers.
CATEGORICAL_FEATURES = [
    "Animal Type", "Intake Type", "Intake Condition", "Sex upon Intake", "Breed", "Color",
]
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
    if factor is None or n < 0:  # a handful of intakes record ages like "-1 years"
        return None
    return n * factor


def load_year(year: int) -> pd.DataFrame:
    """Load the stays that started in ``year``, with engineered features."""
    path = BY_YEAR_DIR / f"{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset for year {year}: {path}")
    df = pd.read_parquet(path)
    df["age_days"] = df["Age upon Intake"].apply(_age_to_days)
    return df


def load_years(years: Iterable[int]) -> pd.DataFrame:
    """Concatenate multiple years into one frame."""
    frames = [load_year(y) for y in years]
    return pd.concat(frames, ignore_index=True)


ID_COLUMN = "Animal ID"


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows missing required features/target and return (X, y)."""
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    clean = df.dropna(subset=needed)
    return clean[FEATURE_COLUMNS].copy(), clean[TARGET_COLUMN].copy()


def split_xy_id(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Like ``split_xy`` but also return the per-row ``Animal ID``.

    The id is needed by the A/B flow to assign each row to a variant. Rows are
    filtered on exactly the same columns as ``split_xy`` so the (X, y) pair is
    identical to what training/robustness see -- the only addition is the id
    column carried alongside.
    """
    needed = FEATURE_COLUMNS + [TARGET_COLUMN]
    clean = df.dropna(subset=needed)
    return (clean[FEATURE_COLUMNS].copy(),
            clean[TARGET_COLUMN].copy(),
            clean[ID_COLUMN].astype(str).copy())
