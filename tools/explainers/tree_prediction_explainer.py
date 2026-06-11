"""
tree_prediction_explainer.py

Converts decision paths into
human-readable explanations.
"""


class TreePredictionExplainer:

    def explain(
        self,
        decision_path_result
    ):
        """
        Generate explanation
        from decision path.
        """

        rules = decision_path_result["path"]

        explanation = []

        explanation.append(
            "The model reached its prediction "
            "using the following decision path:"
        )

        for idx, rule in enumerate(
            rules,
            start=1
        ):

            explanation.append(
                f"{idx}. {rule['condition']}"
            )

        explanation.append(
            f"\nLeaf Node: "
            f"{decision_path_result['leaf_node']}"
        )

        return "\n".join(
            explanation
        )