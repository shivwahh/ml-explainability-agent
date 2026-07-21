import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tools.explainers.counterfactual_explainer import (
    CounterfactualExplainer,
)


def _threshold_dataset():
    """A dataset whose label is driven solely by the first feature."""
    rng = np.random.RandomState(0)
    features = rng.uniform(0, 10, size=(300, 2))
    labels = (features[:, 0] >= 5).astype(int)
    return features, labels, ["f0", "f1"]


def _tree():
    features, labels, names = _threshold_dataset()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        features, labels
    )
    return model, names


def _forest():
    features, labels, names = _threshold_dataset()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        features, labels
    )
    return model, names


def test_flips_prediction_by_changing_the_driving_feature():
    model, names = _tree()
    sample = np.array([3.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=1)

    assert str(result["baseline_prediction"]) == "0"
    assert result["already_target"] is False
    assert result["achievable"] is True
    assert result["recommendation"]["feature"] == "f0"
    assert result["recommendation"]["new_value"] > sample[0]


def test_reports_already_target_without_changes():
    model, names = _tree()
    sample = np.array([9.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=1)

    assert result["already_target"] is True
    assert result["achievable"] is True
    assert result["recommendation"] is None


def test_unreachable_target_is_not_achievable():
    model, names = _tree()
    sample = np.array([3.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=99)

    assert result["achievable"] is False
    assert result["recommendation"] is None
    assert result["options"] == []


def test_recommendation_is_the_smallest_change():
    model, names = _tree()
    sample = np.array([3.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=1)
    deltas = [abs(option["delta"]) for option in result["options"]]

    assert deltas == sorted(deltas)


def test_forest_counterfactual_uses_ensemble_thresholds():
    model, names = _forest()
    sample = np.array([3.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=1)

    assert result["achievable"] is True
    assert result["recommendation"]["feature"] == "f0"


def test_xgboost_counterfactual():
    xgboost = pytest.importorskip("xgboost")

    features, labels, names = _threshold_dataset()
    model = xgboost.XGBClassifier(
        n_estimators=12,
        max_depth=3,
        random_state=0,
        tree_method="hist",
    )
    model.fit(features, labels)
    sample = np.array([3.0, 1.0])

    result = CounterfactualExplainer(model, names).explain(sample, target=1)

    assert result["achievable"] is True
    assert result["recommendation"]["feature"] == "f0"
