"""
decision_path_extractor.py

Extracts the exact path followed by a record
inside a trained Decision Tree.
"""

from sklearn.tree import _tree


class DecisionPathExtractor:

    def __init__(
        self,
        model,
        feature_names
    ):
        self.model = model
        self.feature_names = feature_names
        self.tree = model.tree_

    def extract_path(
        self,
        sample
    ):
        """
        Extract decision path
        for a single observation.
        """

        node_indicator = (
            self.model.decision_path(
                sample.reshape(1, -1)
            )
        )

        leaf_id = (
            self.model.apply(
                sample.reshape(1, -1)
            )[0]
        )

        path_rules = []

        node_index = (
            node_indicator.indices[
                node_indicator.indptr[0]:
                node_indicator.indptr[1]
            ]
        )

        for node_id in node_index:

            if node_id == leaf_id:
                continue

            feature_idx = (
                self.tree.feature[node_id]
            )

            threshold = (
                self.tree.threshold[node_id]
            )

            feature_name = (
                self.feature_names[
                    feature_idx
                ]
            )

            sample_value = (
                sample[feature_idx]
            )

            if sample_value <= threshold:

                condition = (
                    f"{feature_name} <= "
                    f"{threshold:.4f}"
                )

            else:

                condition = (
                    f"{feature_name} > "
                    f"{threshold:.4f}"
                )

            path_rules.append(
                {
                    "node": node_id,
                    "feature": feature_name,
                    "value": float(
                        sample_value
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "condition": condition
                }
            )

        return {
            "leaf_node": int(
                leaf_id
            ),
            "path": path_rules
        }