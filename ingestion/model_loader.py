"""
model_loader.py

Ingestion loader for trained model artifacts.

Loads ``.pkl`` and ``.joblib`` model files and extracts normalized
metadata (model type, task type, hyperparameters, feature names and
classes) without making any assumptions about a specific project.
"""

from pathlib import Path
import pickle

import joblib

try:
    from sklearn.base import is_classifier, is_regressor
except ImportError:  # pragma: no cover - sklearn is a core dependency
    is_classifier = None
    is_regressor = None


class ModelArtifactLoader:
    """
    Load a model artifact and extract normalized metadata.

    The loader supports pickle (``.pkl``) and joblib (``.joblib``)
    serialized models. When the artifact is a scikit-learn ``Pipeline``,
    the final estimator is used for metadata extraction.
    """

    SUPPORTED_EXTENSIONS = {".pkl", ".joblib"}

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the serialized model artifact.

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the file extension is not supported.
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        suffix = self.model_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported model artifact extension: '{suffix}'. "
                f"Supported extensions: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        self.model = None

    def load(self):
        """
        Load and return the deserialized model object.

        Returns:
            The deserialized model object.
        """
        suffix = self.model_path.suffix.lower()

        if suffix == ".joblib":
            self.model = joblib.load(self.model_path)
        else:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)

        return self.model

    def _get_estimator(self):
        """
        Return the underlying estimator, unwrapping a Pipeline.

        Returns:
            The final estimator when the model is a scikit-learn
            ``Pipeline``, otherwise the model itself.
        """
        model = self.model

        if hasattr(model, "steps") and model.steps:
            return model.steps[-1][1]

        return model

    def _infer_task_type(self, estimator) -> str:
        """
        Infer whether the estimator is a classifier or regressor.

        Returns:
            One of ``"classification"``, ``"regression"`` or
            ``"unknown"``.
        """
        if is_classifier is not None and is_classifier(estimator):
            return "classification"

        if is_regressor is not None and is_regressor(estimator):
            return "regression"

        if hasattr(estimator, "classes_"):
            return "classification"

        return "unknown"

    def _extract_hyperparameters(self, estimator) -> dict:
        """
        Extract estimator hyperparameters when available.
        """
        get_params = getattr(estimator, "get_params", None)

        if callable(get_params):
            return get_params()

        return {}

    def _extract_feature_names(self) -> list:
        """
        Extract feature names from the fitted model when available.
        """
        feature_names = getattr(
            self.model,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            estimator = self._get_estimator()
            feature_names = getattr(
                estimator,
                "feature_names_in_",
                None,
            )

        if feature_names is None:
            return []

        return [str(name) for name in feature_names]

    def _extract_classes(self, estimator) -> list:
        """
        Extract class labels for classifiers when available.
        """
        classes = getattr(estimator, "classes_", None)

        if classes is None:
            return []

        return [
            item.item() if hasattr(item, "item") else item
            for item in classes
        ]

    def _extract_n_features(self, estimator) -> int:
        """
        Extract the number of input features when available.
        """
        n_features = getattr(self.model, "n_features_in_", None)

        if n_features is None:
            n_features = getattr(estimator, "n_features_in_", None)

        return int(n_features) if n_features is not None else 0

    def extract_metadata(self) -> dict:
        """
        Extract normalized metadata from the loaded model.

        The model is loaded automatically if it has not been loaded yet.

        Returns:
            A dictionary describing the model artifact.
        """
        if self.model is None:
            self.load()

        estimator = self._get_estimator()

        return {
            "source_path": str(self.model_path),
            "model_type": type(estimator).__name__,
            "is_pipeline": hasattr(self.model, "steps"),
            "task_type": self._infer_task_type(estimator),
            "n_features": self._extract_n_features(estimator),
            "feature_names": self._extract_feature_names(),
            "classes": self._extract_classes(estimator),
            "hyperparameters": self._extract_hyperparameters(estimator),
        }
