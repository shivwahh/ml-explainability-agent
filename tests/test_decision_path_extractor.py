from sklearn.datasets import (
    load_breast_cancer
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)


def test_extract_path():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    extractor = (
        DecisionPathExtractor(
            model,
            data.feature_names
        )
    )

    result = (
        extractor.extract_path(
            X[0]
        )
    )

    assert (
        len(
            result["path"]
        ) > 0
    )