"""
xgboost_path_extractor.py

Decision-path explainer for XGBoost gradient-boosted models.

An XGBoost model has no single scikit-learn ``tree_`` and does not
expose ``estimators_``; its structure lives inside the underlying
``Booster``. This extractor reads every boosted tree from the booster,
traces the branch a sample follows through each one, then aggregates how
often each feature is used across the ensemble.

It exposes the same :meth:`extract_path` entry point and returns the same
ensemble-shaped result as
:class:`tools.tree_reader.forest_path_extractor.ForestPathExtractor`, so
the two are interchangeable behind the explainer selector.
"""

import math


class XGBoostPathExtractor:
    """
    Trace and aggregate decision paths across an XGBoost booster's trees.

    Args:
        model: An XGBoost sklearn-API estimator or a native booster.
        feature_names: Ordered feature names for the model inputs.
    """

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names or [])

    def _booster(self):
        """
        Return the underlying booster for sklearn-API or native models.
        """
        get_booster = getattr(self.model, "get_booster", None)

        if callable(get_booster):
            return get_booster()

        return self.model

    def _value_map(self, sample):
        """
        Map both the real feature name and its ``f{index}`` alias to the
        sample value, matching however the booster labelled its splits.
        """
        values = {}

        for index, name in enumerate(self.feature_names):
            value = float(sample[index])
            values[name] = value
            values[f"f{index}"] = value

        return values

    def extract_path(self, sample):
        """
        Trace and aggregate the decision path across every boosted tree.

        Args:
            sample: A single observation as a 1-D array.

        Returns:
            A dictionary describing the ensemble path with the keys:

            * ``is_ensemble`` - always ``True``.
            * ``n_estimators`` - number of boosted trees traced.
            * ``estimator_paths`` - per-tree ``{leaf_node, path}`` results.
            * ``feature_usage`` - features ranked by how many trees used
              them, as ``{feature, tree_count}`` records.
        """
        frame = self._booster().trees_to_dataframe()
        values = self._value_map(sample)

        nodes = {row["ID"]: row for _, row in frame.iterrows()}
        roots = frame[frame["Node"] == 0].sort_values("Tree")

        estimator_paths = []
        usage = {}

        for _, root in roots.iterrows():
            single = self._trace_tree(root["ID"], nodes, values)
            estimator_paths.append(single)

            for feature in {rule["feature"] for rule in single["path"]}:
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
            "n_estimators": len(estimator_paths),
            "estimator_paths": estimator_paths,
            "feature_usage": feature_usage,
        }

    def _trace_tree(self, node_id, nodes, values):
        """
        Follow one boosted tree from its root to a leaf for the sample.

        XGBoost sends ``value < split`` to the ``Yes`` branch, everything
        else to ``No``, and any missing value to the ``Missing`` branch.
        """
        path_rules = []
        current = nodes.get(node_id)

        while current is not None and current["Feature"] != "Leaf":
            feature = current["Feature"]
            threshold = float(current["Split"])
            value = values.get(feature)

            if value is None or math.isnan(value):
                next_id = current["Missing"]
                condition = f"{feature} is missing"
            elif value < threshold:
                next_id = current["Yes"]
                condition = f"{feature} < {threshold:.4f}"
            else:
                next_id = current["No"]
                condition = f"{feature} >= {threshold:.4f}"

            path_rules.append(
                {
                    "node": current["ID"],
                    "feature": feature,
                    "value": None if value is None else float(value),
                    "threshold": threshold,
                    "condition": condition,
                }
            )

            current = nodes.get(next_id)

        leaf_id = current["ID"] if current is not None else node_id

        return {
            "leaf_node": leaf_id,
            "path": path_rules,
        }
