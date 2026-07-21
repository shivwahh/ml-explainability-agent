"""
catboost_adapter.py

Adapter for CatBoost gradient-boosted models.

Handles the scikit-learn API wrappers (CatBoostClassifier / CatBoostRegressor),
normalizing framework, task type, estimator count and feature names on top of the
base ModelAdapter fields.

The adapter recognizes CatBoost purely from the estimator's module path, so
importing this module never requires catboost to be installed.
"""

from ingestion.adapters.base import ModelAdapter


class CatBoostAdapter(ModelAdapter):
    """
    Adapter for CatBoost estimators.

    Matches any object whose class is defined in the ``catboost`` package.
    """

    name = "catboost"

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Return whether the estimator comes from the catboost package.
        """
        module = type(estimator).__module__ or ""
        return module == "catboost" or module.startswith("catboost.")

    def extract(self, model, estimator, source_path) -> dict:
        """
        Extract normalized metadata plus catboost-specific details.
        """
        metadata = super().extract(model, estimator, source_path)
        metadata["framework"] = "catboost"
        metadata["booster_details"] = self._booster_details(estimator)

        if not metadata["feature_names"]:
            metadata["feature_names"] = self._native_feature_names(estimator)

        if not metadata["n_features"] and metadata["feature_names"]:
            metadata["n_features"] = len(metadata["feature_names"])

        return metadata

    def _booster_details(self, estimator) -> dict:
        """
        Extract ensemble details (tree count) for the model.
        """
        details = {"is_ensemble": True}

        # Check for tree_count_ attribute
        tree_count = getattr(estimator, "tree_count_", None)
        if tree_count is not None:
            details["n_estimators"] = int(tree_count)
        else:
            # Fallback to get_all_params() if available
            get_all_params = getattr(estimator, "get_all_params", None)
            if callable(get_all_params):
                try:
                    params = get_all_params()
                    iterations = params.get("iterations")
                    if iterations is not None:
                        details["n_estimators"] = int(iterations)
                except Exception:  # pragma: no cover
                    pass

        return details

    def _native_feature_names(self, estimator) -> list:
        """
        Extract feature names carried on the estimator.
        """
        names = getattr(estimator, "feature_names_", None)
        if not names:
            return []

        return [str(name) for name in names]
