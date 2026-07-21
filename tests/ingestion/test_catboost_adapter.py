import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_iris

from ingestion.adapters import (
    GenericModelAdapter,
    CatBoostAdapter,
    default_registry,
)

catboost = pytest.importorskip("catboost")


def _fitted_classifier():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    model = catboost.CatBoostClassifier(
        iterations=5,
        depth=3,
        random_seed=42,
        verbose=False,
    )
    model.fit(X, data.target)
    return model, list(data.feature_names)


def test_registry_resolves_catboost_sklearn_api():
    model, _ = _fitted_classifier()

    adapter = default_registry().resolve(model)

    assert isinstance(adapter, CatBoostAdapter)


def test_catboost_adapter_normalizes_metadata():
    model, names = _fitted_classifier()

    metadata = CatBoostAdapter().extract(model, model, "model.joblib")

    assert metadata["adapter"] == "catboost"
    assert metadata["framework"] == "catboost"
    assert metadata["task_type"] == "classification"
    assert metadata["n_features"] == len(names)
    assert metadata["booster_details"]["is_ensemble"] is True
    assert metadata["booster_details"]["n_estimators"] == 5


def test_catboost_adapter_reads_native_feature_names():
    model, names = _fitted_classifier()

    metadata = CatBoostAdapter().extract(model, model, "model.joblib")

    assert metadata["feature_names"] == names
    assert metadata["n_features"] == len(names)
