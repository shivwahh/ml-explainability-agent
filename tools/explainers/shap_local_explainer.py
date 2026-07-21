"""
shap_local_explainer.py

Computes per-feature contributions for a single sample using SHAP.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class ShapLocalExplainer:
    """
    Computes local feature attribution (contributions) using SHAP.

    Supports scikit-learn tree-based models and ensembles, as well as XGBoost
    models (both scikit-learn API and native Booster).
    """

    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize the explainer.

        Args:
            model: Fitted estimator or native model.
            feature_names: List of feature names matching model inputs.
        """
        self.model = model
        self.feature_names = list(feature_names or [])

    def explain_instance(self, sample: Any, prediction: Optional[Any] = None) -> Dict[str, Any]:
        """
        Explain a single instance using SHAP.

        Args:
            sample: 1-D array-like containing feature values.
            prediction: Optional prediction label or value to align multiclass SHAP output.

        Returns:
            A dictionary containing:
                - base_value: expected model output.
                - prediction_value: base_value + sum of contributions.
                - contributions: list of dicts with keys 'feature', 'shap_value', 'feature_value',
                                sorted by absolute shap_value descending.
        """
        import shap  # imported lazily and defensively

        sample_arr = np.asarray(sample, dtype=float)
        sample_2d = sample_arr.reshape(1, -1)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(sample_2d)

        # Handle multiclass or multi-output SHAP shape
        # shap_values could be:
        # 1. list of length n_classes, each of shape (n_samples, n_features)
        # 2. 3D array of shape (n_samples, n_features, n_classes)
        # 3. 2D array of shape (n_samples, n_features)

        class_idx = 0
        is_multiclass = False

        if isinstance(shap_values, list):
            is_multiclass = True
            n_classes = len(shap_values)
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            is_multiclass = True
            n_classes = shap_values.shape[2]
        else:
            n_classes = 1

        if is_multiclass:
            # Find class index matching the prediction
            if prediction is not None and hasattr(self.model, "classes_") and self.model.classes_ is not None:
                pred_str = str(prediction)
                for idx, cls in enumerate(self.model.classes_):
                    if str(cls) == pred_str:
                        class_idx = idx
                        break
            elif n_classes == 2:
                # Default to positive class for binary classification
                class_idx = 1
            else:
                class_idx = 0

        # Extract selected SHAP values for the sample
        if isinstance(shap_values, list):
            selected_shap = shap_values[class_idx][0]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            selected_shap = shap_values[0, :, class_idx]
        else:
            selected_shap = shap_values[0]

        # Extract expected value
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            if len(expected) > class_idx:
                base_value = float(expected[class_idx])
            else:
                base_value = float(expected[0])
        else:
            base_value = float(expected)

        contributions = []
        for idx, name in enumerate(self.feature_names):
            val = float(selected_shap[idx]) if idx < len(selected_shap) else 0.0
            feat_val = float(sample_arr[idx]) if idx < len(sample_arr) else 0.0
            contributions.append({
                "feature": name,
                "shap_value": val,
                "feature_value": feat_val
            })

        # Sort contributions by absolute value descending
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        prediction_value = base_value + sum(c["shap_value"] for c in contributions)

        return {
            "base_value": base_value,
            "prediction_value": prediction_value,
            "contributions": contributions
        }
