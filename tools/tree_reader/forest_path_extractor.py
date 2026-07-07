"""
forest_path_extractor.py

Extract an aggregated decision-path explanation from a fitted tree
ensemble (random forest / extra-trees).

A forest has no single ``tree_``; instead this extractor walks the path
of every estimator for a sample, then aggregates how often each feature
is used across the ensemble so the explanation stays readable.
"""

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)


class ForestPathExtractor:
    """
    Aggregate per-estimator decision paths for a tree ensemble.

    Exposes the same :meth:`extract_path` entry point as
    :class:`DecisionPathExtractor` so the two are interchangeable behind
    the explainer selector.
    """

    def __init__(self, model, feature_names):
        """
        Args:
            model: A fitted forest exposing ``estimators_``.
            feature_names: Ordered feature names for the model.
        """
        self.model = model
        self.feature_names = feature_names

    def extract_path(self, sample):
        """
        Extract and aggregate the decision path across all estimators.

        Args:
            sample: A single observation as a 1-D array.

        Returns:
            A dictionary describing the ensemble path with the keys:

            * ``is_ensemble`` - always ``True``.
            * ``n_estimators`` - number of trees in the forest.
            * ``estimator_paths`` - per-tree ``{leaf_node, path}`` results.
            * ``feature_usage`` - features ranked by how many trees used
              them, as ``{feature, tree_count}`` records.
        """
        estimators = self.model.estimators_

        estimator_paths = []
        usage = {}

        for estimator in estimators:
            single = DecisionPathExtractor(
                estimator,
                self.feature_names,
            ).extract_path(sample)

            estimator_paths.append(single)

            features_in_tree = {
                rule["feature"] for rule in single["path"]
            }

            for feature in features_in_tree:
                usage[feature] = usage.get(feature, 0) + 1

        feature_usage = [
            {"feature": feature, "tree_count": count}
            for feature, count in sorted(
                usage.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        return {
            "is_ensemble": True,
            "n_estimators": len(estimators),
            "estimator_paths": estimator_paths,
            "feature_usage": feature_usage,
        }
