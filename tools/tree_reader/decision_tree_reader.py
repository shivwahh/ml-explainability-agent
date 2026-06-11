"""
decision_tree_reader.py

Utilities for understanding and
explaining Decision Tree models.
"""

from sklearn.tree import _tree


class DecisionTreeReader:

    def __init__(self, model):
        self.model = model
        self.tree = model.tree_

    def get_tree_summary(self):
        """
        Return basic tree information.
        """

        return {
            "node_count": self.tree.node_count,
            "max_depth": self.tree.max_depth,
            "n_features": self.model.n_features_in_
        }

    def get_feature_name(
        self,
        feature_index,
        feature_names
    ):

        if feature_index == _tree.TREE_UNDEFINED:
            return None

        return feature_names[feature_index]

    def extract_rules(
        self,
        feature_names
    ):
        """
        Extract all split rules.
        """

        rules = []

        for node_id in range(
            self.tree.node_count
        ):

            feature_idx = (
                self.tree.feature[node_id]
            )

            if (
                feature_idx
                != _tree.TREE_UNDEFINED
            ):

                feature_name = (
                    feature_names[
                        feature_idx
                    ]
                )

                threshold = (
                    self.tree.threshold[
                        node_id
                    ]
                )

                rules.append(
                    {
                        "node": node_id,
                        "feature": feature_name,
                        "threshold": threshold
                    }
                )

        return rules

    def get_leaf_nodes(self):

        leaves = []

        for node_id in range(
            self.tree.node_count
        ):

            left = self.tree.children_left[
                node_id
            ]

            right = self.tree.children_right[
                node_id
            ]

            if left == right:

                leaves.append(
                    node_id
                )

        return leaves