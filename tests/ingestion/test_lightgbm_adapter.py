import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_iris

from ingestion.adapters import (
    GenericModelAdapter,
    LightGBMAdapter,
    default_registry,
)

lightgbm = pytest.importorskip("lightgbm")


def _fitted_classifier():
    data = load_iris()
    names = [name.replace(" ", "_") for name in data.feature_names]
    X = pd.DataFrame(data.data, columns=names)
    model = lightgbm.LGBMClassifier(
        n_estimators=8,
        max_depth=3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, data.target)
    return model, names


def test_registry_resolves_lightgbm_sklearn_api():
    model, _ = _fitted_classifier()

    adapter = default_registry().resolve(model)

    assert isinstance(adapter, LightGBMAdapter)


def test_lightgbm_adapter_normalizes_metadata():
    model, names = _fitted_classifier()

    metadata = LightGBMAdapter().extract(model, model, "model.joblib")

    assert metadata["adapter"] == "lightgbm"
    assert metadata["framework"] == "lightgbm"
    assert metadata["task_type"] == "classification"
    assert metadata["n_features"] == len(names)
    assert metadata["booster_details"]["is_ensemble"] is True
    assert metadata["booster_details"]["n_estimators"] == 8


def test_lightgbm_adapter_reads_native_feature_names():
    model, names = _fitted_classifier()

    metadata = LightGBMAdapter().extract(model, model, "model.joblib")

    assert metadata["feature_names"] == names
    assert metadata["n_features"] == len(names)
