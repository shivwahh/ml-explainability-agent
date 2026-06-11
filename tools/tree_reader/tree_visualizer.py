"""
tree_visualizer.py

Visualize Decision Tree models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class TreeVisualizer:

    def __init__(
        self,
        model,
        feature_names,
        class_names
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    def display(
        self,
        figsize=(20, 10)
    ):
        """
        Display decision tree.
        """

        plt.figure(
            figsize=figsize
        )

        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )

        plt.tight_layout()

        plt.show()

    def save(
        self,
        output_path
    ):
        """
        Save tree image.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(
            figsize=(20, 10)
        )

        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            bbox_inches="tight"
        )

        plt.close()

        return output_path