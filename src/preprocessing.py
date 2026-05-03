"""
Stateless preprocessing helpers used inside the training pipeline.

Lives in its own module (rather than inline inside ``training.py``) because
training is also executable as ``python -m src.training``, which would set
``__name__ == "__main__"`` and bake the wrong qualified path into any helpers
stored in a serialized pipeline. Keeping helpers here guarantees their import
path is stable as ``src.preprocessing.<helper>`` regardless of how they were
first reached.
"""
from __future__ import annotations


def to_string_lists(X):
    """Convert a DataFrame of categoricals into the ``[[\"col=val\", ...], ...]``
    shape that ``FeatureHasher(input_type='string')`` expects.
    """
    return [[f"{c}={v}" for c, v in zip(X.columns, row)] for row in X.values]
