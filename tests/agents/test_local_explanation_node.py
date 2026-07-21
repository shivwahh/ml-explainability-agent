import numpy as np
from sklearn.tree import DecisionTreeClassifier
from unittest import mock

from agents.nodes import local_explanation_node


def _tree_state():
    rng = np.random.RandomState(0)
    features = rng.uniform(0, 10, size=(200, 2))
    labels = (features[:, 0] >= 5).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        features, labels
    )

    return {
        "model": model,
        "knowledge": {"model": {"feature_names": ["f0", "f1"]}},
        "sample": np.array([3.0, 1.0]),
    }


def test_local_explanation_node_populates_result():
    result = local_explanation_node(_tree_state())

    local_exp = result["local_explanation"]
    assert "base_value" in local_exp
    assert "prediction_value" in local_exp
    assert isinstance(local_exp["contributions"], list)
    assert len(local_exp["contributions"]) == 2
    assert result["prediction"] == "0"


def test_local_explanation_node_degrades_gracefully_without_shap():
    with mock.patch.dict("sys.modules", {"shap": None}):
        state = _tree_state()
        result = local_explanation_node(state)
        # Should return state unchanged (or without local_explanation populated)
        assert "local_explanation" not in result
