"""
xgboost_feature_importance.py

Feature-importance explainer for XGBoost models.

Exposes the same ``get_feature_importance`` / ``get_top_features``
interface as :class:`FeatureImportanceExtractor` but reads importance
from the underlying booster (gain-based by default), so it works for
both the sklearn-API estimators and native ``xgboost.Booster`` objects.
"""

import pandas as pd


class XGBoostFeatureImportanceExtractor:
    """
    Extract gain/weight based feature importance from XGBoost models.

    Args:
        model: An XGBoost sklearn-API estimator or a native booster.
        feature_names: Ordered feature names for the model inputs.
        importance_type: Booster importance type (``"gain"``,
            ``"weight"``, ``"cover"``, ``"total_gain"`` or
            ``"total_cover"``).
    """

    def __init__(self, model, feature_names, importance_type="gain"):
        self.model = model
        self.feature_names = list(feature_names or [])
        self.importance_type = importance_type

    def _booster(self):
        """
        Return the underlying booster for sklearn-API or native models.
        """
        get_booster = getattr(self.model, "get_booster", None)

        if callable(get_booster):
            return get_booster()

        return self.model

    def _scores(self) -> dict:
        """
        Return the raw booster importance scores keyed by feature.
        """
        booster = self._booster()
        get_score = getattr(booster, "get_score", None)

        if not callable(get_score):
            return {}

        try:
            return get_score(importance_type=self.importance_type)
        except Exception:  # pragma: no cover - unfitted booster
            return {}

    def get_feature_importance(self, sort_descending=True):
        """
        Return a feature-importance dataframe.

        Features that never appear in a split are reported with an
        importance of ``0.0``.
        """
        scores = self._scores()

        importances = []
        for index, name in enumerate(self.feature_names):
            value = scores.get(name)

            if value is None:
                value = scores.get(f"f{index}", 0.0)

            importances.append(float(value))

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        )

        if sort_descending:
            importance_df = (
                importance_df
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )

        return importance_df

    def get_top_features(self, top_n=10):
        """
        Return the top ``top_n`` most important features.
        """
        return self.get_feature_importance().head(top_n)
