"""
Pre-training data quality test: attribute distributions.

We test the distributions of two categorical attributes against expectations
derived from domain knowledge about the Austin Animal Center.

Attribute 1: Animal Type
-------------------------
Expectation: All values belong to the set {Dog, Cat, Bird, Other, Livestock},
and Dogs plus Cats together make up at least 90% of records.

Rationale: The Austin Animal Center is a municipal shelter that primarily
serves domestic pets. National shelter intake statistics from the ASPCA
consistently show that dogs and cats account for the overwhelming majority of
animals entering shelters. The dataset itself confirms five distinct animal
types. A drop below 90% for dogs and cats combined would indicate either a data
quality issue (miscategorized animals, new unexpected categories) or a
fundamental change in shelter operations that warrants investigation.

Attribute 2: Outcome Type
--------------------------
Expectation: All values belong to a known set of outcomes, and the Adoption
rate falls between 30% and 60% of all records.

Rationale: The dataset documentation states that Austin is the largest "No
Kill" city in the US, with over 90% of animals adopted, transferred, or
returned to their owners. Adoption is the single largest outcome category but
does not account for all positive outcomes, since transfers to partner rescues
are also significant. From the observed data, the adoption rate is approximately
49%. We set bounds of 30% to 60% to accommodate year over year variation while
flagging any major anomaly. A rate below 30% would contradict Austin's no kill
status, while above 60% would suggest transfers are being misclassified.
"""
import great_expectations as gx

EXPECTED_ANIMAL_TYPES = ["Dog", "Cat", "Bird", "Other", "Livestock"]

EXPECTED_OUTCOME_TYPES = [
    "Adoption", "Transfer", "Return to Owner", "Euthanasia", "Died",
    "Rto-Adopt", "Disposal", "Missing", "Relocate", "Stolen", "Lost",
]


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
    """Dogs and Cats together should make up at least 90% of records."""
    total = len(df)
    dogs_cats = df["Animal Type"].isin(["Dog", "Cat"]).sum()
    ratio = dogs_cats / total
    if ratio < 0.90:
        inspect_rows(~df["Animal Type"].isin(["Dog", "Cat"]), "not Dog or Cat")
    assert ratio >= 0.90, f"Dogs+Cats ratio is {ratio:.2%}, expected >= 90%"


# --- Outcome Type distribution ---

# Halucinated DOGSHT
def test_outcome_type_values_in_expected_set(validate):
    """Outcome Type values should belong to the known set (allowing up to 1% unknown for future additions)."""
    result = validate(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Outcome Type", value_set=EXPECTED_OUTCOME_TYPES, mostly=0.99
        )
    )
    assert result.success, f"Unexpected Outcome Type values: {result.result}"


def test_outcome_type_adoption_rate_in_range(df, inspect_rows):
    """Adoption rate should be between 30% and 60%, consistent with Austin's no kill policy."""
    non_null = df["Outcome Type"].dropna()
    adoption_rate = (non_null == "Adoption").sum() / len(non_null)
    if not (0.30 <= adoption_rate <= 0.60):
        inspect_rows(df["Outcome Type"] == "Adoption", "adoptions")
    assert 0.30 <= adoption_rate <= 0.60, (
        f"Adoption rate is {adoption_rate:.2%}, expected between 30% and 60%"
    )
