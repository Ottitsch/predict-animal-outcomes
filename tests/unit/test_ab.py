"""Unit tests for the A/B split and the significance logic."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from predict_animal_outcomes.ab_test import _macro_f1, assign_bucket, compare

CLASSES = np.array(["Adoption", "Transfer", "Return to Owner", "Euthanasia"])


def _rows(y_true, y_pred, test_id="t"):
    ids = [f"A{i}" for i in range(len(y_true))]
    return pd.DataFrame({"animal_id": ids,
                         "bucket": [assign_bucket(i, test_id) for i in ids],
                         "y_true": y_true, "y_pred": y_pred})


def test_bucket_is_deterministic_and_salted():
    assert assign_bucket("A123", "exp1") == assign_bucket("A123", "exp1")
    ids = [f"A{i}" for i in range(2000)]
    a = np.array([assign_bucket(i, "exp1") for i in ids])
    b = np.array([assign_bucket(i, "exp2") for i in ids])
    assert 0.45 < a.mean() < 0.55
    assert 0.4 < (a == b).mean() < 0.6  # different salts give unrelated splits


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 5, size=300)
        p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 5, size=300))
        p[p == 4] = 3  # a class that is never predicted
        assert np.isclose(_macro_f1(y, p, 5), f1_score(y, p, average="macro", zero_division=0))


def test_identical_models_are_not_significantly_different():
    rng = np.random.default_rng(2)
    y = rng.choice(CLASSES, size=4000, p=[0.6, 0.25, 0.1, 0.05])
    pred = np.where(rng.random(4000) < 0.7, y, rng.choice(CLASSES, size=4000))
    result = compare(_rows(y, pred), _rows(y, pred), "t", n_boot=300)
    assert result["paired"]["rows_where_versions_disagree"] == 0
    assert result["decision_by_paired_macro_f1"] == "no significant difference"


def test_clearly_better_model_wins():
    rng = np.random.default_rng(3)
    y = rng.choice(CLASSES, size=4000, p=[0.6, 0.25, 0.1, 0.05])
    weak = np.where(rng.random(4000) < 0.5, y, rng.choice(CLASSES, size=4000))
    strong = np.where(rng.random(4000) < 0.9, y, rng.choice(CLASSES, size=4000))
    result = compare(_rows(y, weak), _rows(y, strong), "t", n_boot=300)
    lo, hi = result["paired"]["macro_f1_b_minus_a_ci95"]
    assert 0 < lo < hi
    assert result["decision_by_paired_macro_f1"] == "B"
    flipped = compare(_rows(y, strong), _rows(y, weak), "t", n_boot=300)
    assert flipped["decision_by_paired_macro_f1"] == "A"


def test_paired_interval_is_narrower_than_split_interval():
    rng = np.random.default_rng(4)
    y = rng.choice(CLASSES, size=6000, p=[0.6, 0.25, 0.1, 0.05])
    a = np.where(rng.random(6000) < 0.7, y, rng.choice(CLASSES, size=6000))
    b = a.copy()
    flip = rng.random(6000) < 0.02  # B differs from A on about 2% of rows
    b[flip] = rng.choice(CLASSES, size=int(flip.sum()))
    r = compare(_rows(y, a), _rows(y, b), "t", n_boot=300)
    width = lambda ci: ci[1] - ci[0]
    assert width(r["paired"]["macro_f1_b_minus_a_ci95"]) < width(r["split"]["macro_f1_b_minus_a_ci95"]) / 3
