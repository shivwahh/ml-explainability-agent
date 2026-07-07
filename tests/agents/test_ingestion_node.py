import joblib
from sklearn.tree import DecisionTreeClassifier

from agents.ingestion_node import ingestion_node
from agents.state import ExplainabilityState


def _make_model():
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    y = [0, 1, 1, 0]
    model.fit(X, y)
    return model


def _write_config(tmp_path, model_path=None):
    config_path = tmp_path / "project_config.yaml"
    model_line = (
        f'  model_path: "{model_path}"' if model_path else "  model_path: null"
    )
    config_path.write_text(
        "ingestion:\n"
        f"{model_line}\n"
        "  data_dictionary_path: null\n"
        "  project_context_path: null\n",
        encoding="utf-8",
    )
    return config_path


def test_ingestion_node_populates_knowledge_and_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_make_model(), model_path)

    config_path = _write_config(tmp_path, model_path.as_posix())

    state: ExplainabilityState = {"config_path": str(config_path)}
    result = ingestion_node(state)

    assert result["knowledge"]["model"]["model_type"] == (
        "DecisionTreeClassifier"
    )
    assert result["knowledge"]["validation"]["summary"]["has_model"] is True
    assert result["model"] is not None
    assert hasattr(result["model"], "predict")


def test_ingestion_node_preserves_existing_state(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_make_model(), model_path)

    config_path = _write_config(tmp_path, model_path.as_posix())

    state: ExplainabilityState = {
        "config_path": str(config_path),
        "question": "Why this prediction?",
    }
    result = ingestion_node(state)

    assert result["question"] == "Why this prediction?"


def test_ingestion_node_handles_missing_artifacts(tmp_path):
    config_path = _write_config(tmp_path, model_path=None)

    state: ExplainabilityState = {"config_path": str(config_path)}
    result = ingestion_node(state)

    assert result["model"] is None
    assert result["knowledge"]["model"] is None
    assert result["knowledge"]["validation"]["valid"] is False
