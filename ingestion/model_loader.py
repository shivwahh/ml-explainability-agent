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

from ingestion.adapters import default_registry


class ModelArtifactLoader:
    """
    Load a model artifact and extract normalized metadata.

    The loader supports pickle (``.pkl``) and joblib (``.joblib``)
    serialized models. When the artifact is a scikit-learn ``Pipeline``,
    the final estimator is used for metadata extraction. Framework-specific
    normalization is delegated to a resolved :class:`ModelAdapter`.
    """

    SUPPORTED_EXTENSIONS = {".pkl", ".joblib"}

    def __init__(self, model_path: str, registry=None):
        """
        Args:
            model_path: Path to the serialized model artifact.
            registry: Optional :class:`AdapterRegistry` used to resolve the
                model adapter. A default registry is created when omitted.

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

        self.registry = registry or default_registry()
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

    def extract_metadata(self) -> dict:
        """
        Extract normalized metadata from the loaded model.

        The model is loaded automatically if it has not been loaded yet.
        Framework-specific normalization is delegated to the adapter
        resolved from the registry.

        Returns:
            A dictionary describing the model artifact.
        """
        if self.model is None:
            self.load()

        estimator = self._get_estimator()
        adapter = self.registry.resolve(estimator)

        return adapter.extract(self.model, estimator, self.model_path)
