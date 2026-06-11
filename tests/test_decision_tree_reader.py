from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from tools.tree_reader.decision_tree_reader import (
    DecisionTreeReader
)


def test_tree_summary():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    reader = DecisionTreeReader(
        model
    )

    summary = (
        reader.get_tree_summary()
    )

    assert (
        summary["max_depth"] <= 3
    )