"""
counterfactual_explainer.py

Counterfactual ("what-if") explainer for tree-based models.

Answers the question *"which single feature do I need to change, and to
what value, to flip the model's prediction to a target class?"* by
searching one feature at a time.

Candidate values are drawn from the model's own split thresholds
(scikit-learn trees, forests / extra-trees, and XGBoost boosters), so the
search targets the exact decision boundaries the model uses instead of an
arbitrary grid. Optional ``feature_bounds`` add a bounded linear sweep,
and a relative sweep around the current value is used as a last resort for
any feature the model never splits on.

Pure functionality: no LLM calls.
"""

import numpy as np


class CounterfactualExplainer:
    """
    Search for the minimal single-feature change that flips a prediction.

    Args:
        model: A fitted estimator exposing ``predict``.
        feature_names: Ordered feature names for the model inputs.
        feature_bounds: Optional ``{name: (low, high)}`` map adding a
            bounded linear sweep of candidate values per feature.
        steps: Number of points in the bounded/relative sweeps.
        epsilon: Small offset used to land just either side of a split
            threshold.
    """

    def __init__(
        self,
        model,
        feature_names,
        feature_bounds=None,
        steps=25,
        epsilon=1e-3,
    ):
        self.model = model
        self.feature_names = list(feature_names or [])
        self.feature_bounds = feature_bounds or {}
        self.steps = steps
        self.epsilon = epsilon

    def explain(self, sample, target=1):
        """
        Find the smallest single-feature change that yields ``target``.

        Args:
            sample: A single observation as a 1-D array.
            target: The desired prediction to reach.

        Returns:
            A dictionary with the keys:

            * ``target`` - the requested prediction.
            * ``baseline_prediction`` - the model's current prediction.
            * ``already_target`` - whether the sample already predicts
              ``target``.
            * ``achievable`` - whether a single-feature change reaches
              ``target``.
            * ``recommendation`` - the smallest-magnitude change as
              ``{feature, feature_index, original_value, new_value,
              delta}``, or ``None``.
            * ``options`` - the minimal flip found per feature, sorted by
              absolute change.
        """
        sample = np.asarray(sample, dtype=float)
        baseline = self._predict(sample)

        if self._matches(baseline, target):
            return {
                "target": target,
                "baseline_prediction": baseline,
                "already_target": True,
                "achievable": True,
                "recommendation": None,
                "options": [],
            }

        thresholds = self._thresholds()
        options = []

        for index, name in enumerate(self.feature_names):
            current = float(sample[index])

            for candidate in self._candidate_values(
                index, name, current, thresholds
            ):
                trial = sample.copy()
                trial[index] = candidate

                if self._matches(self._predict(trial), target):
                    options.append(
                        {
                            "feature": name,
                            "feature_index": index,
                            "original_value": round(current, 6),
                            "new_value": round(float(candidate), 6),
                            "delta": round(float(candidate) - current, 6),
                        }
                    )
                    break

        options.sort(key=lambda option: abs(option["delta"]))

        return {
            "target": target,
            "baseline_prediction": baseline,
            "already_target": False,
            "achievable": bool(options),
            "recommendation": options[0] if options else None,
            "options": options,
        }

    def _predict(self, sample):
        """
        Return the model's prediction for a single sample.
        """
        return self.model.predict(sample.reshape(1, -1))[0]

    @staticmethod
    def _matches(prediction, target) -> bool:
        """
        Compare a prediction to the target, tolerating numpy scalar types.
        """
        return str(prediction) == str(target)

    def _candidate_values(self, index, name, current, thresholds):
        """
        Build the ordered candidate values to try for one feature.

        Candidates come from split thresholds (just either side of each),
        an optional bounded sweep, and a relative fallback sweep, sorted
        by how far they move the feature from its current value.
        """
        candidates = set()

        for threshold in thresholds.get(index, []):
            candidates.add(threshold - self.epsilon)
            candidates.add(threshold + self.epsilon)

        if name in self.feature_bounds:
            low, high = self.feature_bounds[name]
            candidates.update(
                np.linspace(low, high, self.steps).tolist()
            )

        if not candidates:
            base = abs(current) if current else 1.0
            candidates.update(
                (current + step * base)
                for step in np.linspace(-2.0, 2.0, self.steps)
            )

        return sorted(
            (c for c in candidates if abs(c - current) > 1e-12),
            key=lambda c: abs(c - current),
        )

    def _thresholds(self) -> dict:
        """
        Map feature index to the split thresholds the model uses.

        Supports scikit-learn single trees, tree ensembles exposing
        ``estimators_``, and XGBoost sklearn-API/native boosters. Returns
        an empty map for models with no inspectable tree structure.
        """
        model = self.model

        if hasattr(model, "tree_"):
            return self._sklearn_tree_thresholds(model.tree_)

        if hasattr(model, "estimators_"):
            merged = {}
            for estimator in np.ravel(model.estimators_):
                tree = getattr(estimator, "tree_", None)
                if tree is None:
                    continue
                for feature, values in self._sklearn_tree_thresholds(
                    tree
                ).items():
                    merged.setdefault(feature, set()).update(values)
            return {feature: sorted(values) for feature, values in merged.items()}

        if self._is_xgboost(model):
            return self._xgboost_thresholds()

        return {}

    @staticmethod
    def _sklearn_tree_thresholds(tree) -> dict:
        """
        Group a scikit-learn tree's thresholds by feature index.

        Leaf nodes carry a negative feature index and are skipped.
        """
        result = {}
        for node in range(tree.node_count):
            feature = int(tree.feature[node])
            if feature < 0:
                continue
            result.setdefault(feature, set()).add(float(tree.threshold[node]))
        return {feature: sorted(values) for feature, values in result.items()}

    @staticmethod
    def _is_xgboost(model) -> bool:
        """
        Return whether the model is an XGBoost estimator or booster.
        """
        if callable(getattr(model, "get_booster", None)):
            return True
        module = type(model).__module__ or ""
        return module == "xgboost" or module.startswith("xgboost.")

    def _xgboost_thresholds(self) -> dict:
        """
        Group an XGBoost booster's split values by feature index.
        """
        get_booster = getattr(self.model, "get_booster", None)
        booster = get_booster() if callable(get_booster) else self.model
        frame = booster.trees_to_dataframe()

        name_to_index = {
            name: index for index, name in enumerate(self.feature_names)
        }

        result = {}
        for _, row in frame.iterrows():
            feature = row["Feature"]
            if feature == "Leaf":
                continue

            index = name_to_index.get(feature)
            if index is None and isinstance(feature, str) and feature.startswith("f"):
                try:
                    index = int(feature[1:])
                except ValueError:
                    index = None

            if index is None:
                continue

            result.setdefault(index, set()).add(float(row["Split"]))

        return {feature: sorted(values) for feature, values in result.items()}
