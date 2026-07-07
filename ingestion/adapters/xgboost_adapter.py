"""
xgboost_adapter.py

Adapter for XGBoost gradient-boosted models.

Handles both the scikit-learn API wrappers (``XGBClassifier`` /
``XGBRegressor`` and their random-forest variants) and native
``xgboost.Booster`` objects, normalizing framework, task type,
estimator count and feature names on top of the base
:class:`ModelAdapter` fields.

The adapter recognizes XGBoost purely from the estimator's module path,
so importing this module never requires xgboost to be installed.
"""

from ingestion.adapters.base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    """
    Adapter for XGBoost estimators.

    Matches any object whose class is defined in the ``xgboost`` package,
    covering the sklearn-API estimators and the native ``Booster``.
    """

    name = "xgboost"

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Return whether the estimator comes from the xgboost package.
        """
        module = type(estimator).__module__ or ""

        return module == "xgboost" or module.startswith("xgboost.")

    def extract(self, model, estimator, source_path) -> dict:
        """
        Extract normalized metadata plus booster-specific structure.
        """
        metadata = super().extract(model, estimator, source_path)
        metadata["framework"] = "xgboost"
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

        n_estimators = getattr(estimator, "n_estimators", None)

        if n_estimators is not None:
            details["n_estimators"] = int(n_estimators)

        booster = self._resolve_booster(estimator)
        num_rounds = getattr(booster, "num_boosted_rounds", None)

        if callable(num_rounds):
            try:
                details["n_estimators"] = int(num_rounds())
            except Exception:  # pragma: no cover - defensive
                pass

        return details

    def _resolve_booster(self, estimator):
        """
        Return the underlying booster for sklearn-API or native models.
        """
        get_booster = getattr(estimator, "get_booster", None)

        if callable(get_booster):
            try:
                return get_booster()
            except Exception:  # pragma: no cover - unfitted model
                return None

        return estimator

    def _native_feature_names(self, estimator) -> list:
        """
        Extract feature names carried on a native booster.
        """
        booster = self._resolve_booster(estimator)
        names = getattr(booster, "feature_names", None)

        if not names:
            return []

        return [str(name) for name in names]
