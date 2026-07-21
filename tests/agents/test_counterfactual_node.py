import numpy as np

from sklearn.tree import DecisionTreeClassifier

from agents.nodes import counterfactual_node


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


def test_counterfactual_node_populates_result():
    result = counterfactual_node(_tree_state())

    counterfactual = result["counterfactual"]

    assert counterfactual["achievable"] is True
    assert counterfactual["recommendation"]["feature"] == "f0"
    # Baseline prediction is surfaced for the explanation prompt.
    assert result["prediction"] == "0"


def test_counterfactual_node_respects_target_override():
    state = {**_tree_state(), "counterfactual_target": 99}

    result = counterfactual_node(state)

    assert result["counterfactual"]["achievable"] is False
    assert result["counterfactual"]["recommendation"] is None
