"""
Shared pytest fixtures for data quality tests.
"""
import pandas as pd
import great_expectations as gx
import pytest
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.parquet"


@pytest.fixture(scope="session")
def df():
    """Load the parquet dataset as a plain pandas DataFrame."""
    return pd.read_parquet(DATA_PATH)


@pytest.fixture(scope="session")
def gx_batch(df):
    """Create a Great Expectations batch from the dataset for running expectations."""
    context = gx.get_context()
    ds = context.data_sources.add_pandas("pandas")
    asset = ds.add_dataframe_asset("data")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    return batch_def.get_batch(batch_parameters={"dataframe": df})
