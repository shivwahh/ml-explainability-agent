import joblib
import pytest
from sklearn.tree import DecisionTreeClassifier

from ingestion.model_loader import ModelArtifactLoader


def _make_model():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 1, 0]
    model.fit(X, y)
    return model


def test_load_joblib_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_make_model(), model_path)

    loader = ModelArtifactLoader(str(model_path))
    loaded = loader.load()

    assert isinstance(loaded, DecisionTreeClassifier)


def test_load_pickle_model(tmp_path):
    import pickle

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(_make_model(), file)

    loader = ModelArtifactLoader(str(model_path))
    loaded = loader.load()

    assert isinstance(loaded, DecisionTreeClassifier)


def test_extract_metadata(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_make_model(), model_path)

    metadata = ModelArtifactLoader(str(model_path)).extract_metadata()

    assert metadata["model_type"] == "DecisionTreeClassifier"
    assert metadata["task_type"] == "classification"
    assert metadata["n_features"] == 2
    assert metadata["classes"] == [0, 1]
    assert "max_depth" in metadata["hyperparameters"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ModelArtifactLoader("does_not_exist.joblib")


def test_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "model.txt"
    bad_path.write_text("not a model", encoding="utf-8")

    with pytest.raises(ValueError):
        ModelArtifactLoader(str(bad_path))
