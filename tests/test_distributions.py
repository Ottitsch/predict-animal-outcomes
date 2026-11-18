"""
Data quality test: attribute distributions.

We test the distributions of categorical attributes against expectations
derived from domain knowledge about the Austin Animal Center. Rates are checked
per intake year, because a pooled rate can hide a single broken year.

Attribute 1: Animal Type
-------------------------
Expectation: All values belong to the set {Dog, Cat, Bird, Other, Livestock},
and Dogs plus Cats together make up at least 90% of stays in every year.

Rationale: The Austin Animal Center is a municipal shelter that primarily
serves domestic pets. National shelter intake statistics from the ASPCA
consistently show that dogs and cats account for the overwhelming majority of
animals entering shelters. The dataset itself confirms five distinct animal
types. A drop below 90% for dogs and cats combined would indicate either a data
quality issue (miscategorized animals, new unexpected categories) or a
fundamental change in shelter operations that warrants investigation. Observed:
93% to 95% in every year from 2014 to 2024.

Attribute 2: Intake Type
-------------------------
Expectation: All values belong to the six intake types the shelter records.
A new value would be a new category the one hot encoder has never seen, which
the model silently ignores, so we want to hear about it.

Attribute 3: Outcome Type
--------------------------
Expectation: All values belong to a known set of outcomes, and the Adoption
share of each year's stays falls between 30% and 75%.

Rationale: The dataset documentation states that Austin is the largest "No
Kill" city in the US, with over 90% of animals adopted, transferred, or
returned to their owners. Adoption is the single largest outcome category but
does not account for all positive outcomes, since transfers to partner rescues
are also significant. The adoption share is not stable, though: it rose
steadily from 38% of 2014 stays to 62% of 2024 stays while transfers fell from
33% to 22%. The first version of this test pooled all years and allowed 30% to
60%, which hid that recent years are above 60% (2024 is at 62%). The bounds
now sit about ten points outside the observed range: below 30% would
contradict Austin's no kill status, and above 75% would suggest transfers or
returns are being recorded as adoptions.
"""
import great_expectations as gx

EXPECTED_ANIMAL_TYPES = ["Dog", "Cat", "Bird", "Other", "Livestock"]

EXPECTED_INTAKE_TYPES = [
    "Stray", "Owner Surrender", "Public Assist", "Wildlife", "Abandoned",
    "Euthanasia Request",
]

EXPECTED_OUTCOME_TYPES = [
    "Adoption", "Transfer", "Return to Owner", "Euthanasia", "Died",
    "Rto-Adopt", "Disposal", "Missing", "Relocate", "Stolen", "Lost",
]


def _per_year(df):
    return df.groupby(df["intake_time"].dt.year)


# --- Animal Type distribution ---

def test_animal_type_values_in_expected_set(validate):
    """Every Animal Type value must be one of the five known categories."""
    result = validate(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Animal Type", value_set=EXPECTED_ANIMAL_TYPES
        )
    )
    assert result.success, f"Unexpected Animal Type values: {result.result}"


def test_animal_type_dogs_and_cats_dominate(df, inspect_rows):
    """Dogs and Cats together should make up at least 90% of each year's stays."""
    ratio = _per_year(df)["Animal Type"].apply(lambda s: s.isin(["Dog", "Cat"]).mean())
    low = ratio[ratio < 0.90]
    if not low.empty:
        inspect_rows(~df["Animal Type"].isin(["Dog", "Cat"]), "not Dog or Cat")
    assert low.empty, f"Dogs+Cats share below 90% in: {low.round(3).to_dict()}"


# --- Intake Type ---

def test_intake_type_values_in_expected_set(validate):
    """Every Intake Type value must be one of the six known intake types."""
    result = validate(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Intake Type", value_set=EXPECTED_INTAKE_TYPES
        )
    )
    assert result.success, f"Unexpected Intake Type values: {result.result}"


# --- Outcome Type distribution ---

def test_outcome_type_values_in_expected_set(validate):
    """Every Outcome Type value must be one of the eleven known outcomes."""
    result = validate(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Outcome Type", value_set=EXPECTED_OUTCOME_TYPES
        )
    )
    assert result.success, f"Unexpected Outcome Type values: {result.result}"


def test_outcome_type_adoption_rate_in_range(df, inspect_rows):
    """Each year's adoption share should be between 30% and 75%."""
    rate = _per_year(df)["Outcome Type"].apply(lambda s: (s == "Adoption").mean())
    bad = rate[(rate < 0.30) | (rate > 0.75)]
    if not bad.empty:
        inspect_rows(df["Outcome Type"] == "Adoption", "adoptions")
    assert bad.empty, f"Adoption share outside 30% to 75% in: {bad.round(3).to_dict()}"
