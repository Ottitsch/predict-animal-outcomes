"""
Reproduce the v1 versus v2 comparison in the README.

v1 trained on the Outcomes file alone, split by outcome year. v2 trains on
stays (intake joined to outcome) with intake time features, split by intake
year. Same model, same years (train 2014 to 2023, evaluate 2024).

    python data/prepare_data.py        # once, builds the stay files
    python analysis/leakage_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict_animal_outcomes import data as data_mod  # noqa: E402
from predict_animal_outcomes.training import _build_pipeline  # noqa: E402

TRAIN_YEARS = list(range(2014, 2024))
HOLDOUT_YEAR = 2024
V1_CATEGORICAL = ["Animal Type", "Sex upon Outcome", "Breed", "Color"]


def _evaluate(train: pd.DataFrame, holdout: pd.DataFrame, categorical: list[str]) -> dict:
    cols = categorical + ["age_days"]
    train = train.dropna(subset=cols + ["Outcome Type"])
    holdout = holdout.dropna(subset=cols + ["Outcome Type"])
    pipe = _build_pipeline(categorical_features=categorical)
    pipe.fit(train[cols], train["Outcome Type"])
    pred = pipe.predict(holdout[cols])
    y = holdout["Outcome Type"]
    majority = (y == y.mode()[0]).mean()
    acc = accuracy_score(y, pred)
    return {"holdout accuracy": acc, "majority baseline": majority,
            "margin (points)": 100 * (acc - majority),
            "macro F1": f1_score(y, pred, average="macro", zero_division=0)}


def main() -> None:
    # v1: outcome rows, outcome time columns, outcome year split.
    raw = pd.read_csv(ROOT / "data" / "raw" /
                      "Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv")
    raw = raw.dropna(subset=["Outcome Type"])
    raw["year"] = pd.to_datetime(raw["DateTime"].str[:19], format="%Y-%m-%dT%H:%M:%S").dt.year
    raw["age_days"] = raw["Age upon Outcome"].apply(data_mod._age_to_days)
    v1_train, v1_hold = raw[raw.year.isin(TRAIN_YEARS)], raw[raw.year == HOLDOUT_YEAR]

    rows = {
        "v1: Sex upon Outcome, Age upon Outcome, type, breed, color":
            _evaluate(v1_train, v1_hold, V1_CATEGORICAL),
        "v1 without Sex upon Outcome":
            _evaluate(v1_train, v1_hold, [c for c in V1_CATEGORICAL if c != "Sex upon Outcome"]),
        "v2: intake time features (stays)":
            _evaluate(data_mod.load_years(TRAIN_YEARS), data_mod.load_year(HOLDOUT_YEAR),
                      data_mod.CATEGORICAL_FEATURES),
    }
    table = pd.DataFrame(rows).T
    print(table.round(3).to_string())

    adoption = (raw["Outcome Type"] == "Adoption").groupby(raw["Sex upon Outcome"]).mean()
    print("\nAdoption share by Sex upon Outcome (all years):")
    print(adoption.round(3).to_string())


if __name__ == "__main__":
    main()
