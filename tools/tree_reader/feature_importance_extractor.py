"""
feature_importance_extractor.py

Extract feature importance from
Decision Tree models.
"""

import pandas as pd


class FeatureImportanceExtractor:

    def __init__(
        self,
        model,
        feature_names
    ):
        self.model = model
        self.feature_names = feature_names

    def get_feature_importance(
        self,
        sort_descending=True
    ):
        """
        Return feature importance dataframe.
        """

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_
            }
        )

        if sort_descending:
            importance_df = (
                importance_df
                .sort_values(
                    by="importance",
                    ascending=False
                )
                .reset_index(drop=True)
            )

        return importance_df

    def get_top_features(
        self,
        top_n=10
    ):
        """
        Return top N important features.
        """

        return (
            self
            .get_feature_importance()
            .head(top_n)
        )