from sklearn.tree import DecisionTreeClassifier

from tools.model_loader.model_loader import (
    ModelLoader
)


def test_save_and_load_model():

    loader = ModelLoader("models")

    model = DecisionTreeClassifier()

    loader.save_model(
        model,
        "test_model"
    )

    loaded_model = loader.load_model(
        "test_model"
    )

    assert isinstance(
        loaded_model,
        DecisionTreeClassifier
    )