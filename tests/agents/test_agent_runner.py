import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from agents.agent_runner import build_initial_state, run_agent


def _write_model(tmp_path):
    data = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data.data, data.target)
    model.feature_names_in_ = data.feature_names

    model_path = tmp_path / "iris.joblib"
    joblib.dump(model, model_path)
    return model_path


def _write_config(tmp_path):
    model_path = _write_model(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "ingestion:\n"
        f'  model_path: "{model_path.as_posix()}"\n'
        "  data_dictionary_path: null\n"
        "  project_context_path: null\n"
        "routing:\n"
        '  default_intent: "full_explanation"\n'
        "  intents:\n"
        "    feature_importance:\n"
        '      - "which features"\n'
        "    prediction:\n"
        '      - "predict"\n'
        "explanation:\n"
        '  provider: "echo"\n'
        '  model: "gpt-4.1-mini"\n',
        encoding="utf-8",
    )
    return config_path


def test_build_initial_state_converts_sample():
    state = build_initial_state("cfg.yaml", "why?", sample=[1.0, 2.0])

    assert state["config_path"] == "cfg.yaml"
    assert state["question"] == "why?"
    assert list(state["sample"]) == [1.0, 2.0]


def test_build_initial_state_without_sample():
    state = build_initial_state("cfg.yaml", "why?")

    assert "sample" not in state


def test_run_agent_full_pipeline_produces_explanation(tmp_path):
    config_path = _write_config(tmp_path)

    result = run_agent(
        str(config_path),
        "Please give an overview of this result.",
        sample=[5.1, 3.5, 1.4, 0.2],
    )

    assert result["intent"] == "full_explanation"
    assert "echo provider" in result["explanation"]
    assert "prediction" in result


def test_run_agent_feature_importance_branch(tmp_path):
    config_path = _write_config(tmp_path)

    result = run_agent(
        str(config_path),
        "Which features matter most?",
    )

    assert result["intent"] == "feature_importance"
    assert isinstance(result["feature_importance"], list)
    assert "explanation" not in result
