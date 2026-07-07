from agents.explanation_provider import (
    EchoExplanationProvider,
    OpenAIExplanationProvider,
    build_provider,
)
from agents.nodes import explanation_node
from utils.config_loader import ConfigLoader


def _write_config(tmp_path, provider="echo", model="gpt-4.1-mini"):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "explanation:\n"
        f'  provider: "{provider}"\n'
        f'  model: "{model}"\n',
        encoding="utf-8",
    )
    return config_path


def test_build_provider_echo(tmp_path):
    config = ConfigLoader(str(_write_config(tmp_path, provider="echo")))

    provider = build_provider(config)

    assert isinstance(provider, EchoExplanationProvider)


def test_build_provider_openai_without_calling(tmp_path):
    config = ConfigLoader(str(_write_config(tmp_path, provider="openai")))

    provider = build_provider(config)

    assert isinstance(provider, OpenAIExplanationProvider)
    assert provider.model == "gpt-4.1-mini"


def test_build_provider_unknown_falls_back_to_echo(tmp_path):
    config = ConfigLoader(str(_write_config(tmp_path, provider="mystery")))

    provider = build_provider(config)

    assert isinstance(provider, EchoExplanationProvider)


def test_build_provider_reads_model_from_config(tmp_path):
    config = ConfigLoader(
        str(_write_config(tmp_path, provider="openai", model="gpt-4o"))
    )

    provider = build_provider(config)

    assert provider.model == "gpt-4o"


def test_echo_provider_includes_prompt():
    text = EchoExplanationProvider().generate("hello prompt")

    assert "hello prompt" in text


def test_explanation_node_uses_echo_provider(tmp_path):
    config_path = _write_config(tmp_path, provider="echo")

    state = {
        "config_path": str(config_path),
        "question": "Why this class?",
        "prediction": "0",
        "decision_path": {"leaf_node": 1, "path": []},
        "feature_importance": [{"feature": "petal length (cm)", "importance": 0.6}],
    }

    result = explanation_node(state)

    assert isinstance(result["explanation"], str)
    assert "Why this class?" in result["explanation"]
    assert "petal length (cm)" in result["explanation"]
