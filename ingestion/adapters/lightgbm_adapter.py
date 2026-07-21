"""
lightgbm_adapter.py

Adapter for LightGBM gradient-boosted models.

Handles the scikit-learn API wrappers (LGBMClassifier / LGBMRegressor),
normalizing framework, task type, estimator count and feature names on top of the
base ModelAdapter fields.

The adapter recognizes LightGBM purely from the estimator's module path, so
importing this module never requires lightgbm to be installed.
"""

from ingestion.adapters.base import ModelAdapter


class LightGBMAdapter(ModelAdapter):
    """
    Adapter for LightGBM estimators.

    Matches any object whose class is defined in the ``lightgbm`` package.
    """

    name = "lightgbm"

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Return whether the estimator comes from the lightgbm package.
        """
        module = type(estimator).__module__ or ""
        return module == "lightgbm" or module.startswith("lightgbm.")

    def extract(self, model, estimator, source_path) -> dict:
        """
        Extract normalized metadata plus lightgbm-specific details.
        """
        metadata = super().extract(model, estimator, source_path)
        metadata["framework"] = "lightgbm"
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

        # Check for n_estimators in the sklearn API wrapper
        n_estimators = getattr(estimator, "n_estimators", None)
        if n_estimators is not None:
            details["n_estimators"] = int(n_estimators)
        else:
            # LightGBM booster exposes num_trees() method
            booster = getattr(estimator, "booster_", None)
            if booster is None:
                booster = estimator
            if booster is not None and hasattr(booster, "num_trees"):
                num_trees = getattr(booster, "num_trees", None)
                if callable(num_trees):
                    try:
                        details["n_estimators"] = int(num_trees())
                    except Exception:  # pragma: no cover
                        pass

        return details

    def _native_feature_names(self, estimator) -> list:
        """
        Extract feature names carried on the estimator.
        """
        names = getattr(estimator, "feature_name_", None)
        if not names:
            return []

        return [str(name) for name in names]
