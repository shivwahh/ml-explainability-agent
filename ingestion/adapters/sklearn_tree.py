"""
sklearn_tree.py

Adapter for scikit-learn decision tree and forest models.

Handles single trees (``DecisionTreeClassifier`` /
``DecisionTreeRegressor``) and tree ensembles (random forests and
extra-trees), adding tree-specific structural metadata on top of the
normalized fields from :class:`ModelAdapter`.
"""

from ingestion.adapters.base import ModelAdapter


class SklearnTreeAdapter(ModelAdapter):
    """
    Adapter for scikit-learn tree-based estimators.

    Matches single decision trees (estimators exposing a fitted
    ``tree_``) and forest ensembles of trees (estimators whose class is a
    known forest type exposing ``estimators_``).
    """

    name = "sklearn_tree"

    _FOREST_TYPES = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
    }

    @classmethod
    def matches(cls, estimator) -> bool:
        """
        Return whether the estimator is an sklearn tree or forest.
        """
        if hasattr(estimator, "tree_"):
            return True

        is_forest = type(estimator).__name__ in cls._FOREST_TYPES

        return is_forest and hasattr(estimator, "estimators_")

    def extract(self, model, estimator, source_path) -> dict:
        """
        Extract normalized metadata plus tree-specific structure.
        """
        metadata = super().extract(model, estimator, source_path)
        metadata["framework"] = "scikit-learn"
        metadata["tree_details"] = self._tree_details(estimator)

        return metadata

    def _tree_details(self, estimator) -> dict:
        """
        Extract structural details for a single tree or a forest.
        """
        if hasattr(estimator, "estimators_"):
            return {
                "is_ensemble": True,
                "n_estimators": len(estimator.estimators_),
            }

        details = {"is_ensemble": False}

        get_depth = getattr(estimator, "get_depth", None)

        if callable(get_depth):
            details["depth"] = int(get_depth())

        get_n_leaves = getattr(estimator, "get_n_leaves", None)

        if callable(get_n_leaves):
            details["n_leaves"] = int(get_n_leaves())

        tree = getattr(estimator, "tree_", None)

        if tree is not None and hasattr(tree, "node_count"):
            details["node_count"] = int(tree.node_count)

        return details
