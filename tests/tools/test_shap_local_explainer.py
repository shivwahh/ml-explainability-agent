import numpy as np
import pytest
from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from tools.explainers.shap_local_explainer import ShapLocalExplainer


def test_shap_local_explainer_classifier():
    iris = load_iris()
    X, y, names = iris.data, iris.target, list(iris.feature_names)
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)

    explainer = ShapLocalExplainer(model, names)
    sample = X[0]

    # Run with prediction = 0
    res = explainer.explain_instance(sample, prediction="0")

    assert "base_value" in res
    assert "prediction_value" in res
    assert isinstance(res["contributions"], list)
    assert len(res["contributions"]) == len(names)
    assert res["contributions"][0]["feature"] in names

    # Contributions must be sorted by absolute SHAP value descending
    shap_vals = [c["shap_value"] for c in res["contributions"]]
    abs_shap_vals = [abs(v) for v in shap_vals]
    assert abs_shap_vals == sorted(abs_shap_vals, reverse=True)


def test_shap_local_explainer_regressor():
    diabetes = load_diabetes()
    X, y, names = diabetes.data, diabetes.target, list(diabetes.feature_names)
    model = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)

    explainer = ShapLocalExplainer(model, names)
    sample = X[0]

    res = explainer.explain_instance(sample)

    assert "base_value" in res
    assert "prediction_value" in res
    assert len(res["contributions"]) == len(names)


def test_shap_local_explainer_xgboost():
    xgb = pytest.importorskip("xgboost")
    iris = load_iris()
    X, y, names = iris.data, iris.target, list(iris.feature_names)
    model = xgb.XGBClassifier(n_estimators=3, max_depth=3, random_state=42).fit(X, y)

    explainer = ShapLocalExplainer(model, names)
    sample = X[0]

    res = explainer.explain_instance(sample, prediction="0")

    assert "base_value" in res
    assert "prediction_value" in res
    assert len(res["contributions"]) == len(names)
