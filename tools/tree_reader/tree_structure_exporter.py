"""
tree_structure_exporter.py

Export Decision Tree structure into JSON format
for Explainability Agents.
"""

from pathlib import Path
import json

import numpy as np
from sklearn.tree import _tree


class TreeStructureExporter:

    def __init__(
        self,
        model,
        feature_names,
        class_names=None
    ):
        self.model = model
        self.tree = model.tree_
        self.feature_names = feature_names
        self.class_names = class_names

    def _to_python(self, value):
        """
        Convert NumPy types to native Python types.
        """

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        return value

    def export_to_json(
        self,
        output_file
    ):
        """
        Export tree structure to JSON.
        """

        tree_data = {
            "model_type": type(self.model).__name__,
            "max_depth": int(self.tree.max_depth),
            "node_count": int(self.tree.node_count),
            "n_features": int(self.model.n_features_in_),
            "nodes": []
        }

        for node_id in range(
            self.tree.node_count
        ):

            feature_idx = self.tree.feature[node_id]

            is_leaf = bool(
                feature_idx ==
                _tree.TREE_UNDEFINED
            )

            node_info = {
                "node_id": int(node_id),
                "is_leaf": is_leaf,
                "samples": int(
                    self.tree.n_node_samples[node_id]
                )
            }

            if is_leaf:

                class_distribution = (
                    self.tree.value[node_id][0]
                )

                class_distribution = [
                    float(v)
                    for v in class_distribution
                ]

                predicted_class = int(
                    np.argmax(
                        class_distribution
                    )
                )

                node_info.update(
                    {
                        "prediction": class_distribution,
                        "predicted_class": predicted_class
                    }
                )

                if self.class_names is not None:

                    node_info[
                        "predicted_class_name"
                    ] = str(
                        self.class_names[
                            predicted_class
                        ]
                    )

            else:

                node_info.update(
                    {
                        "feature": str(
                            self.feature_names[
                                feature_idx
                            ]
                        ),
                        "threshold": float(
                            self.tree.threshold[
                                node_id
                            ]
                        ),
                        "left_child": int(
                            self.tree.children_left[
                                node_id
                            ]
                        ),
                        "right_child": int(
                            self.tree.children_right[
                                node_id
                            ]
                        )
                    }
                )

            tree_data["nodes"].append(
                node_info
            )

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                tree_data,
                f,
                indent=4
            )

        return output_file