"""
feature_importance_visualizer.py
"""

import matplotlib.pyplot as plt


class FeatureImportanceVisualizer:

    def plot(
        self,
        importance_df,
        top_n=10
    ):

        plot_df = (
            importance_df
            .head(top_n)
            .sort_values(
                by="importance"
            )
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            plot_df["feature"],
            plot_df["importance"]
        )

        plt.xlabel(
            "Importance"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Feature Importance"
        )

        plt.tight_layout()

        plt.show()