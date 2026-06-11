"""
config_loader.py

Utility functions for loading project configuration files.
"""

from pathlib import Path
import yaml


class ConfigLoader:
    """
    Loads configuration from YAML files.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Read YAML configuration file.
        """
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get(self, *keys, default=None):
        """
        Retrieve nested configuration values.

        Example:
        config.get("model", "type")
        """
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

        return value if value is not None else default

    def as_dict(self):
        """
        Return complete configuration.
        """
        return self.config