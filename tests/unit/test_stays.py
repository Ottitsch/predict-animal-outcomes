"""Unit tests for how intakes and outcomes are joined into stays."""
import importlib.util
from pathlib import Path

import pandas as pd

from predict_animal_outcomes.data import _age_to_days

_spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[2] / "data" / "prepare_data.py")
prepare_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_data)


def _intake(animal, when, **kw):
    row = {"Animal ID": animal, "Name": None, "DateTime": when, "MonthYear": None,
           "Found Location": "x", "Intake Type": "Stray", "Intake Condition": "Normal",
           "Animal Type": "Dog", "Sex upon Intake": "Intact Male", "Age upon Intake": "1 year",
           "Breed": "Mix", "Color": "Black"}
    row.update(kw)
    return row


def _outcome(animal, when, outcome="Adoption"):
    return {"Animal ID": animal, "Date of Birth": None, "Name": None, "DateTime": when,
            "MonthYear": None, "Outcome Type": outcome, "Outcome Subtype": None,
            "Animal Type": "Dog", "Sex upon Outcome": "Neutered Male",
            "Age upon Outcome": "1 year", "Breed": "Mix", "Color": "Black"}


def _stays(intakes, outcomes):
    stays, counts = prepare_data.build_stays(pd.DataFrame(intakes), pd.DataFrame(outcomes))
    return stays.sort_values(["Animal ID", "intake_time"]).reset_index(drop=True), counts


def test_each_intake_gets_the_first_outcome_after_it():
    stays, _ = _stays(
        [_intake("A1", "01/01/2020 10:00:00 AM"), _intake("A1", "03/01/2020 10:00:00 AM")],
        [_outcome("A1", "2020-02-01T09:00:00", "Return to Owner"),
         _outcome("A1", "2020-03-05T09:00:00", "Adoption")],
    )
    assert stays["Outcome Type"].tolist() == ["Return to Owner", "Adoption"]


def test_outcome_columns_are_not_carried_over():
    stays, _ = _stays([_intake("A1", "01/01/2020 10:00:00 AM")],
                      [_outcome("A1", "2020-01-05T09:00:00")])
    assert "Sex upon Outcome" not in stays.columns
    assert "Age upon Outcome" not in stays.columns
    assert stays.loc[0, "Sex upon Intake"] == "Intact Male"


def test_intake_without_outcome_is_dropped():
    stays, counts = _stays([_intake("A1", "01/01/2020 10:00:00 AM"),
                            _intake("A2", "01/01/2020 10:00:00 AM")],
                           [_outcome("A1", "2020-01-05T09:00:00")])
    assert stays["Animal ID"].tolist() == ["A1"]
    assert counts["intakes_without_outcome"] == 1


def test_outcome_after_a_later_intake_does_not_close_the_earlier_stay():
    # The first stay's outcome record is missing; the only outcome belongs to the
    # second stay and must not be paired with the first one.
    stays, counts = _stays(
        [_intake("A1", "01/01/2020 10:00:00 AM"), _intake("A1", "03/01/2020 10:00:00 AM")],
        [_outcome("A1", "2020-03-05T09:00:00")],
    )
    assert len(stays) == 1
    assert stays.loc[0, "intake_time"] == pd.Timestamp("2020-03-01 10:00")
    assert counts["intakes_whose_next_outcome_follows_a_later_intake"] == 1


def test_outcome_offsets_are_read_as_wall_clock():
    # Same day, outcome two hours after intake. Read as UTC and converted to
    # Austin time, the "-05:00" row would still be right but the naive row
    # would move five hours earlier, before the intake.
    stays, _ = _stays(
        [_intake("A1", "06/01/2020 10:00:00 AM"), _intake("A2", "06/01/2020 10:00:00 AM")],
        [_outcome("A1", "2020-06-01T12:00:00-05:00"), _outcome("A2", "2020-06-01T12:00:00")],
    )
    assert len(stays) == 2
    assert (stays["outcome_time"] == pd.Timestamp("2020-06-01 12:00")).all()


def test_exact_duplicate_rows_are_ignored():
    stays, _ = _stays([_intake("A1", "01/01/2020 10:00:00 AM")] * 2,
                      [_outcome("A1", "2020-01-05T09:00:00")] * 2)
    assert len(stays) == 1


def test_age_parsing():
    assert _age_to_days("2 years") == 730
    assert _age_to_days("3 weeks") == 21
    assert _age_to_days("1 month") == 30
    assert _age_to_days("-1 years") is None
    assert _age_to_days("unknown") is None
    assert _age_to_days(None) is None
