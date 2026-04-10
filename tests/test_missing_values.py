"""
Pre-training data quality test: missing / null values.

We test all columns for null rates against documented expectations.

Expectation rationale
---------------------
The Austin Animal Center assigns every animal a unique Animal ID at intake and
records its Animal Type (Dog, Cat, Bird, Other, Livestock) at the same time.
These two fields are mandatory parts of the intake process, so we expect zero
nulls in both columns.

Outcome Type represents the animal's status when it leaves the center. The
dataset documentation states that over 90% of animals are adopted, transferred,
or returned. Every record in this dataset represents an animal leaving the
center, so an outcome type must always be present. We require zero nulls.

DateTime is the timestamp of the outcome event. Because it is recorded at the
moment the outcome occurs, it must always be present. We require zero nulls.

The remaining columns have relaxed thresholds. Date of Birth, MonthYear, Sex
upon Outcome, Age upon Outcome, Breed, and Color are recorded at intake but
may have gaps for strays or animals with unknown history, so we allow up to 5%
nulls. Name is frequently missing (many shelter animals arrive unnamed), so we
allow up to 50% nulls.

Additional info:
I dont test "Outcome Subtype" because I dont care about it and plan on dropping it.
(I only want to predict "Outcome Type" and "Outcome Subtype" would be too much of 
a hint for the model)
"""
import great_expectations as gx


def test_animal_id_not_null(gx_batch):
    """Animal ID is assigned at intake and must never be null."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Animal ID", mostly=1.0)
    )
    assert result.success, f"Animal ID has unexpected nulls: {result.result}"


def test_animal_type_not_null(gx_batch):
    """Animal Type is recorded at intake and must never be null."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Animal Type", mostly=1.0)
    )
    assert result.success, f"Animal Type has unexpected nulls: {result.result}"


def test_outcome_type_not_null(gx_batch):
    """Outcome Type must never be null."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Outcome Type", mostly=1.0)
    )
    assert result.success, f"Outcome Type has unexpected nulls: {result.result}"


def test_datetime_not_null(gx_batch):
    """DateTime must never be null."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="DateTime", mostly=1.0)
    )
    assert result.success, f"DateTime has unexpected nulls: {result.result}"


# --- Relaxed null checks for remaining columns ---


def test_date_of_birth_mostly_not_null(gx_batch):
    """Date of Birth should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Date of Birth", mostly=0.95)
    )
    assert result.success, f"Date of Birth has >5% nulls: {result.result}"


def test_name_mostly_not_null(gx_batch):
    """Name may be missing for unnamed animals, but should be present for at least 50%."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Name", mostly=0.50)
    )
    assert result.success, f"Name has >50% nulls: {result.result}"


def test_monthyear_mostly_not_null(gx_batch):
    """MonthYear should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthYear", mostly=0.95)
    )
    assert result.success, f"MonthYear has >5% nulls: {result.result}"


def test_sex_upon_outcome_mostly_not_null(gx_batch):
    """Sex upon Outcome should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Sex upon Outcome", mostly=0.95)
    )
    assert result.success, f"Sex upon Outcome has >5% nulls: {result.result}"


def test_age_upon_outcome_mostly_not_null(gx_batch):
    """Age upon Outcome should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Age upon Outcome", mostly=0.95)
    )
    assert result.success, f"Age upon Outcome has >5% nulls: {result.result}"


def test_breed_mostly_not_null(gx_batch):
    """Breed should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Breed", mostly=0.95)
    )
    assert result.success, f"Breed has >5% nulls: {result.result}"


def test_color_mostly_not_null(gx_batch):
    """Color should be present for at least 95% of records."""
    result = gx_batch.validate(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Color", mostly=0.95)
    )
    assert result.success, f"Color has >5% nulls: {result.result}"
