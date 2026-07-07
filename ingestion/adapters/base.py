"""
base.py

Base model adapter interface.

Defines the :class:`ModelAdapter` contract and the shared normalization
helpers (task type, feature names, classes, feature count and
hyperparameters) that concrete framework adapters build upon.
"""

try:
    from sklearn.base import is_classifier, is_regressor
except ImportError:  # pragma: no cover - sklearn is a core dependency
    is_classifier = None
    is_regressor = None


class ModelAdapter:
    """
    Base class for framework-specific model adapters.

    Subclasses declare the framework they support via :meth:`matches`
    and may extend :meth:`extract` to add framework-specific metadata on
    top of the normalized fields produced by the base implementation.
    """

    name = "base"

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Return whether this adapter can handle the given estimator.

        Args:
            estimator: The underlying estimator (a Pipeline is already
                unwrapped to its final estimator by the caller).

        Returns:
            ``True`` when the adapter supports the estimator.
        """
        raise NotImplementedError

    def extract(self, model, estimator, source_path) -> dict:
        """
        Extract normalized metadata for the given model.

        Args:
            model: The full loaded model object (possibly a Pipeline).
            estimator: The unwrapped final estimator.
            source_path: Path the artifact was loaded from.

        Returns:
            A dictionary of normalized model metadata.
        """
        return {
            "source_path": str(source_path),
            "adapter": self.name,
            "model_type": type(estimator).__name__,
            "is_pipeline": hasattr(model, "steps"),
            "task_type": self._infer_task_type(estimator),
            "n_features": self._extract_n_features(model, estimator),
            "feature_names": self._extract_feature_names(model, estimator),
            "classes": self._extract_classes(estimator),
            "hyperparameters": self._extract_hyperparameters(estimator),
        }

    def _infer_task_type(self, estimator) -> str:
        """
        Infer whether the estimator is a classifier or regressor.

        Returns:
            One of ``"classification"``, ``"regression"`` or
            ``"unknown"``.
        """
        if is_classifier is not None:
            try:
                if is_classifier(estimator):
                    return "classification"

                if is_regressor(estimator):
                    return "regression"
            except Exception:
                pass

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

    def _extract_feature_names(self, model, estimator) -> list:
        """
        Extract feature names from the fitted model when available.
        """
        feature_names = getattr(model, "feature_names_in_", None)

        if feature_names is None:
            feature_names = getattr(estimator, "feature_names_in_", None)

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

    def _extract_n_features(self, model, estimator) -> int:
        """
        Extract the number of input features when available.
        """
        n_features = getattr(model, "n_features_in_", None)

        if n_features is None:
            n_features = getattr(estimator, "n_features_in_", None)

        return int(n_features) if n_features is not None else 0
