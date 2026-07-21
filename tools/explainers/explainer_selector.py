"""
explainer_selector.py

Adapter-aware selection of explainability tools.

Chooses the correct explainer for a model based on the normalized
knowledge object's metadata (the adapter's ``tree_details``), falling
back to attribute inspection when metadata is unavailable. This keeps the
agent nodes free of hardcoded model assumptions.
"""

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)
from tools.tree_reader.feature_importance_extractor import (
    FeatureImportanceExtractor
)
from tools.tree_reader.forest_path_extractor import (
    ForestPathExtractor
)
from tools.explainers.xgboost_feature_importance import (
    XGBoostFeatureImportanceExtractor
)
from tools.explainers.xgboost_path_extractor import (
    XGBoostPathExtractor
)


def is_ensemble(model, model_metadata=None) -> bool:
    """
    Return whether the model is a tree ensemble.

    Prefers the normalized ``tree_details.is_ensemble`` flag from the
    knowledge object and falls back to checking for ``estimators_``.
    """
    details = (model_metadata or {}).get("tree_details")

    if isinstance(details, dict) and "is_ensemble" in details:
        return bool(details["is_ensemble"])

    return hasattr(model, "estimators_")


def is_xgboost(model, model_metadata=None) -> bool:
    """
    Return whether the model is an XGBoost estimator.

    Prefers the normalized ``framework``/``adapter`` fields from the
    knowledge object and falls back to inspecting the model's module.
    """
    metadata = model_metadata or {}

    if metadata.get("framework") == "xgboost":
        return True

    if metadata.get("adapter") == "xgboost":
        return True

    module = type(model).__module__ or ""

    return module == "xgboost" or module.startswith("xgboost.")


def select_decision_path_explainer(model, feature_names, model_metadata=None):
    """
    Return the decision-path explainer matching the model.

    An XGBoost model yields an :class:`XGBoostPathExtractor` (it has no
    scikit-learn ``tree_``); a tree ensemble yields a
    :class:`ForestPathExtractor`; a single tree yields a
    :class:`DecisionPathExtractor`. All expose ``extract_path``.
    """
    if is_xgboost(model, model_metadata):
        return XGBoostPathExtractor(model, feature_names)

    if is_ensemble(model, model_metadata):
        return ForestPathExtractor(model, feature_names)

    return DecisionPathExtractor(model, feature_names)


def select_feature_importance_explainer(
    model,
    feature_names,
    model_metadata=None,
):
    """
    Return the feature-importance explainer for the model.

    XGBoost models resolve to a gain-based
    :class:`XGBoostFeatureImportanceExtractor`. Single trees and forests
    both expose ``feature_importances_``, so a single
    :class:`FeatureImportanceExtractor` covers those cases.
    """
    if is_xgboost(model, model_metadata):
        return XGBoostFeatureImportanceExtractor(model, feature_names)

    return FeatureImportanceExtractor(model, feature_names)


def select_local_explainer(model, feature_names, model_metadata=None):
    """
    Return the SHAP local explainer for supported tree/boosted models.

    Degrades gracefully by returning None when shap is not installed.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        return None

    from tools.explainers.shap_local_explainer import ShapLocalExplainer

    return ShapLocalExplainer(model, feature_names)

