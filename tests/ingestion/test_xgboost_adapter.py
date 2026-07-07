import numpy as np
import pytest

from sklearn.datasets import load_iris

from ingestion.adapters import (
    GenericModelAdapter,
    XGBoostAdapter,
    default_registry,
)

xgboost = pytest.importorskip("xgboost")


def _fitted_classifier():
    data = load_iris()
    model = xgboost.XGBClassifier(
        n_estimators=8,
        max_depth=3,
        random_state=42,
        tree_method="hist",
    )
    model.fit(data.data, data.target)
    return model, list(data.feature_names)


def _fitted_native_booster():
    data = load_iris()
    names = list(data.feature_names)
    dtrain = xgboost.DMatrix(data.data, label=data.target, feature_names=names)
    params = {"max_depth": 3, "objective": "multi:softprob", "num_class": 3}
    booster = xgboost.train(params, dtrain, num_boost_round=6)
    return booster, names


def test_registry_resolves_xgboost_sklearn_api():
    model, _ = _fitted_classifier()

    adapter = default_registry().resolve(model)

    assert isinstance(adapter, XGBoostAdapter)


def test_registry_resolves_native_booster():
    booster, _ = _fitted_native_booster()

    adapter = default_registry().resolve(booster)

    assert isinstance(adapter, XGBoostAdapter)


def test_xgboost_adapter_normalizes_metadata():
    model, names = _fitted_classifier()

    metadata = XGBoostAdapter().extract(model, model, "model.json")

    assert metadata["adapter"] == "xgboost"
    assert metadata["framework"] == "xgboost"
    assert metadata["task_type"] == "classification"
    assert metadata["n_features"] == len(names)
    assert metadata["booster_details"]["is_ensemble"] is True
    assert metadata["booster_details"]["n_estimators"] >= 1


def test_xgboost_adapter_reads_native_feature_names():
    booster, names = _fitted_native_booster()

    metadata = XGBoostAdapter().extract(booster, booster, "model.json")

    assert metadata["framework"] == "xgboost"
    assert metadata["feature_names"] == names
    assert metadata["n_features"] == len(names)


def test_non_xgboost_model_does_not_match():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit([[0, 0], [1, 1]], [0, 1])

    adapter = default_registry().resolve(model)

    assert isinstance(adapter, GenericModelAdapter)
