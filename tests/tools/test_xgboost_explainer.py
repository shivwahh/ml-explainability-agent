import numpy as np
import pytest

from sklearn.datasets import load_iris

from tools.explainers.explainer_selector import (
    is_xgboost,
    select_decision_path_explainer,
    select_feature_importance_explainer,
)
from tools.explainers.xgboost_feature_importance import (
    XGBoostFeatureImportanceExtractor,
)
from tools.explainers.xgboost_path_extractor import (
    XGBoostPathExtractor,
)
from tools.tree_reader.feature_importance_extractor import (
    FeatureImportanceExtractor,
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


def test_is_xgboost_from_metadata():
    assert is_xgboost(None, {"framework": "xgboost"}) is True
    assert is_xgboost(None, {"adapter": "xgboost"}) is True
    assert is_xgboost(None, {"framework": "scikit-learn"}) is False


def test_is_xgboost_from_module():
    model, _ = _fitted_classifier()

    assert is_xgboost(model) is True


def test_selector_returns_xgboost_explainer():
    model, names = _fitted_classifier()

    explainer = select_feature_importance_explainer(
        model, names, {"framework": "xgboost"}
    )

    assert isinstance(explainer, XGBoostFeatureImportanceExtractor)


def test_selector_returns_sklearn_explainer_for_tree():
    from sklearn.tree import DecisionTreeClassifier

    data = load_iris()
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(
        data.data, data.target
    )

    explainer = select_feature_importance_explainer(
        tree, list(data.feature_names), {"framework": "scikit-learn"}
    )

    assert isinstance(explainer, FeatureImportanceExtractor)


def test_xgboost_importance_sklearn_api():
    model, names = _fitted_classifier()

    importance = XGBoostFeatureImportanceExtractor(
        model, names
    ).get_top_features(len(names))

    assert list(importance["feature"]) and set(importance["feature"]) <= set(names)
    assert len(importance) == len(names)
    assert (importance["importance"] >= 0).all()
    assert importance["importance"].sum() > 0


def test_xgboost_importance_native_booster():
    booster, names = _fitted_native_booster()

    importance = XGBoostFeatureImportanceExtractor(
        booster, names
    ).get_feature_importance()

    assert len(importance) == len(names)
    assert importance["importance"].sum() > 0


def test_selector_returns_xgboost_path_explainer():
    model, names = _fitted_classifier()

    explainer = select_decision_path_explainer(
        model, names, {"framework": "xgboost"}
    )

    assert isinstance(explainer, XGBoostPathExtractor)


def test_xgboost_path_traces_without_tree_attribute():
    model, names = _fitted_classifier()
    sample = np.array([5.1, 3.5, 1.4, 0.2])

    result = select_decision_path_explainer(
        model, names, {"framework": "xgboost"}
    ).extract_path(sample)

    assert result["is_ensemble"] is True
    assert result["n_estimators"] == len(result["estimator_paths"])
    assert result["n_estimators"] > 0
    assert all("tree_count" in entry for entry in result["feature_usage"])

    conditions = [
        rule["condition"]
        for path in result["estimator_paths"]
        for rule in path["path"]
    ]
    assert conditions


def test_xgboost_path_native_booster():
    booster, names = _fitted_native_booster()
    sample = np.array([6.3, 3.3, 6.0, 2.5])

    result = XGBoostPathExtractor(booster, names).extract_path(sample)

    assert result["n_estimators"] > 0
    assert {entry["feature"] for entry in result["feature_usage"]} <= set(names)
