from utils.config_loader import ConfigLoader


def test_load_config():

    config = ConfigLoader(
        "configs/project_config.yaml"
    )

    assert config.get(
        "project",
        "name"
    ) == "ML Explainability Agent"