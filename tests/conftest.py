"""
Shared pytest fixtures for data quality tests.
"""
import pandas as pd
import great_expectations as gx
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BY_YEAR_DIR = ROOT / "data" / "processed" / "by_year"

_cfg_path = ROOT / "flows" / "training_defaults.yml"
_cfg = yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}


def pytest_addoption(parser):
    parser.addoption(
        "--inspect",
        action="store_true",
        default=False,
        help="On test failure, print the rows that violate the expectation.",
    )
    parser.addoption(
        "--train-years",
        default=str(_cfg.get("train_years", "2014,2015,2016,2017,2018,2019,2020,2021")),
        help="Comma-separated train years (matches flow.py --train_years).",
    )
    parser.addoption(
        "--holdout-year",
        default=str(_cfg.get("holdout_year", 2024)),
        help="Holdout year (matches flow.py --holdout_year).",
    )


@pytest.fixture(scope="session")
def inspect_mode(request):
    return request.config.getoption("--inspect")


@pytest.fixture(scope="session")
def df(request):
    """Load and concatenate the by_year parquets for the run's train + holdout years."""
    train_years = [
        int(y.strip())
        for y in request.config.getoption("--train-years").split(",")
    ]
    holdout_year = int(request.config.getoption("--holdout-year"))
    years = sorted(set(train_years + [holdout_year]))
    frames = [pd.read_parquet(BY_YEAR_DIR / f"{y}.parquet") for y in years]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def gx_batch(df):
    """Create a Great Expectations batch from the dataset for running expectations."""
    context = gx.get_context()
    ds = context.data_sources.add_pandas("pandas")
    asset = ds.add_dataframe_asset("data")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    return batch_def.get_batch(batch_parameters={"dataframe": df})


def _failing_mask(df, expectation):
    """Derive a pandas boolean mask for rows that violate a GX expectation."""
    col = expectation.column
    if isinstance(expectation, gx.expectations.ExpectColumnValuesToNotBeNull):
        return df[col].isna()
    if isinstance(expectation, gx.expectations.ExpectColumnValuesToBeInSet):
        return ~df[col].isin(expectation.value_set) & df[col].notna()
    return pd.Series(False, index=df.index)


@pytest.fixture(scope="session")
def inspect_rows(df, inspect_mode):
    """Print failing rows when --inspect is active. No-op otherwise."""
    def _inspect(mask, label, max_rows=50):
        if not inspect_mode:
            return
        failing = df[mask]
        count = len(failing)
        print(f"\n--- {count} failing rows: {label} (showing up to {max_rows}) ---")
        print(failing.head(max_rows).to_string())
        if count > max_rows:
            print(f"... ({count - max_rows} more rows not shown)")
        print()
    return _inspect


@pytest.fixture(scope="session")
def validate(gx_batch, df, inspect_mode):
    """Wrap gx_batch.validate. On failure with --inspect, prints failing rows."""
    def _validate(expectation, max_rows=50):
        result = gx_batch.validate(expectation)
        if not result.success and inspect_mode:
            mask = _failing_mask(df, expectation)
            failing = df[mask]
            count = len(failing)
            print(f"\n--- {count} failing rows for {expectation.__class__.__name__} "
                  f"on '{expectation.column}' (showing up to {max_rows}) ---")
            print(failing.head(max_rows).to_string())
            if count > max_rows:
                print(f"... ({count - max_rows} more rows not shown)")
            print()
        return result
    return _validate
