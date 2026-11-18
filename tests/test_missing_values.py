"""
Data quality test: missing / null values, and stays that make sense.

These tests run on the per year stay files (``data/processed/by_year``) that
training and evaluation read, one row per intake paired with its outcome.

Expectation rationale
---------------------
The Austin Animal Center assigns every animal a unique Animal ID at intake and
records its Animal Type, Intake Type and Intake Condition at the same time.
These fields are mandatory parts of the intake process, so we expect zero
nulls in all of them, and in the intake time.

A stay only exists once its outcome is known, so Outcome Type and the outcome
time must always be present too.

Sex upon Intake, Age upon Intake, Breed and Color describe the animal and may
have gaps for strays with unknown history, so we allow up to 5% nulls.

Additional info:
I dont test "Outcome Subtype" or anything else recorded at outcome, because the
stay files don't carry it: it would be too much of a hint for the model.
"""
import great_expectations as gx
import pytest

STRICT = ["Animal ID", "Animal Type", "Intake Type", "Intake Condition",
          "intake_time", "outcome_time", "Outcome Type"]
RELAXED = ["Sex upon Intake", "Age upon Intake", "Breed", "Color"]


@pytest.mark.parametrize("column", STRICT)
def test_required_columns_not_null(validate, column):
    """Fields recorded by the intake process, and the outcome, must never be null."""
    result = validate(gx.expectations.ExpectColumnValuesToNotBeNull(column=column, mostly=1.0))
    assert result.success, f"{column} has unexpected nulls: {result.result}"


@pytest.mark.parametrize("column", RELAXED)
def test_descriptive_columns_mostly_not_null(validate, column):
    """Descriptive fields should be present for at least 95% of stays."""
    result = validate(gx.expectations.ExpectColumnValuesToNotBeNull(column=column, mostly=0.95))
    assert result.success, f"{column} has >5% nulls: {result.result}"


def test_no_stay_ends_before_it_starts(df, inspect_rows):
    """Every outcome must come at or after its intake.

    This is the check that catches a timestamp parsing mistake: reading the
    outcome times as UTC once made about 10% of outcomes land hours before
    their own intake.
    """
    bad = df["outcome_time"] < df["intake_time"]
    if bad.any():
        inspect_rows(bad, "outcome before intake")
    assert not bad.any(), f"{int(bad.sum())} stays end before they start"
