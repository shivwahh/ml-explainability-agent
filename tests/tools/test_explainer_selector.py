import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tools.tree_reader.decision_path_extractor import DecisionPathExtractor
from tools.tree_reader.forest_path_extractor import ForestPathExtractor
from tools.tree_reader.feature_importance_extractor import (
    FeatureImportanceExtractor,
)
from tools.explainers.explainer_selector import (
    is_ensemble,
    select_decision_path_explainer,
    select_feature_importance_explainer,
)


def _iris():
    data = load_iris()
    return data.data, data.target, list(data.feature_names)


def _tree():
    X, y, names = _iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    return model, names


def _forest():
    X, y, names = _iris()
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    return model, names


def test_is_ensemble_from_metadata():
    assert is_ensemble(None, {"tree_details": {"is_ensemble": True}}) is True
    assert is_ensemble(None, {"tree_details": {"is_ensemble": False}}) is False


def test_is_ensemble_falls_back_to_attribute():
    forest, _ = _forest()
    tree, _ = _tree()

    assert is_ensemble(forest) is True
    assert is_ensemble(tree) is False


def test_selects_decision_path_for_single_tree():
    tree, names = _tree()

    explainer = select_decision_path_explainer(
        tree, names, {"tree_details": {"is_ensemble": False}}
    )

    assert isinstance(explainer, DecisionPathExtractor)


def test_selects_forest_path_for_ensemble():
    forest, names = _forest()

    explainer = select_decision_path_explainer(
        forest, names, {"tree_details": {"is_ensemble": True}}
    )

    assert isinstance(explainer, ForestPathExtractor)


def test_forest_path_aggregates_across_estimators():
    forest, names = _forest()
    sample = np.array([5.1, 3.5, 1.4, 0.2])

    result = ForestPathExtractor(forest, names).extract_path(sample)

    assert result["is_ensemble"] is True
    assert result["n_estimators"] == 5
    assert len(result["estimator_paths"]) == 5
    assert all(
        "tree_count" in entry for entry in result["feature_usage"]
    )


def test_feature_importance_selector_works_for_both():
    tree, names = _tree()
    forest, _ = _forest()

    tree_explainer = select_feature_importance_explainer(tree, names)
    forest_explainer = select_feature_importance_explainer(forest, names)

    assert isinstance(tree_explainer, FeatureImportanceExtractor)
    assert isinstance(forest_explainer, FeatureImportanceExtractor)
    assert len(tree_explainer.get_top_features(4)) == 4
    assert len(forest_explainer.get_top_features(4)) == 4
