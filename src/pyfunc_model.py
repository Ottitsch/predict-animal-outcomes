"""
MLflow PythonModel wrapper for our skops-serialized scikit-learn pipeline.

We use ``mlflow.pyfunc`` to satisfy MLflow 3.x's logged_model contract while
keeping skops (not pickle) as the actual serialization format for the model.
The wrapper itself is small: at load time it deserialises the skops file and
all subsequent ``predict`` calls delegate to the underlying sklearn pipeline.
"""
from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import skops.io as sio


class AnimalOutcomePyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        skops_path = Path(context.artifacts["skops_model"])
        # ``trusted`` lists the type names we are willing to deserialize.
        # Producing and consuming the artifact happen inside this repo so this
        # is acceptable; skops' design forces it to be an explicit decision.
        self._model = sio.load(
            skops_path,
            trusted=sio.get_untrusted_types(file=skops_path),
        )

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        return self._model.predict(model_input)
