"""
model_loader.py

Utility for saving, loading and inspecting
machine learning model artifacts.
"""

from pathlib import Path
import joblib


class ModelLoader:
    """
    Handles model persistence operations.
    """

    def __init__(self, model_directory: str):

        self.model_directory = Path(model_directory)

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_model(
        self,
        model,
        model_name: str
    ) -> Path:
        """
        Save model to disk.
        """

        model_path = (
            self.model_directory /
            f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        return model_path

    def load_model(
        self,
        model_name: str
    ):
        """
        Load model from disk.
        """

        model_path = (
            self.model_directory /
            f"{model_name}.joblib"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        return joblib.load(model_path)

    def model_exists(
        self,
        model_name: str
    ) -> bool:

        model_path = (
            self.model_directory /
            f"{model_name}.joblib"
        )

        return model_path.exists()

    def list_models(self):
        """
        Return all saved models.
        """

        return [
            file.stem
            for file in self.model_directory.glob(
                "*.joblib"
            )
        ]