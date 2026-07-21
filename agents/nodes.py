from tools.explainers.explainer_selector import (
    select_decision_path_explainer,
    select_feature_importance_explainer,
    select_local_explainer
)

from tools.explainers.counterfactual_explainer import (
    CounterfactualExplainer
)

from agents.explanation_provider import build_provider

from prompts.explanation_prompt import build_explanation_prompt

from utils.config_loader import ConfigLoader


def _feature_names(state):
    """
    Return feature names from the knowledge object.

    Prefers the model's own feature names and falls back to the data
    dictionary feature records.
    """
    knowledge = state.get("knowledge", {})
    model_metadata = knowledge.get("model") or {}

    names = model_metadata.get("feature_names")

    if names:
        return names

    return [
        record.get("name")
        for record in knowledge.get("features", [])
        if record.get("name")
    ]


def _model_metadata(state):
    """
    Return the normalized model metadata from the knowledge object.
    """
    return state.get("knowledge", {}).get("model") or {}


def planner_node(state):

    print("Planner Node")

    return state


def prediction_node(state):

    model = state["model"]

    prediction = model.predict(
        state["sample"].reshape(1, -1)
    )[0]

    return {
        **state,
        "prediction": str(prediction)
    }


def decision_path_node(state):

    explainer = select_decision_path_explainer(
        state["model"],
        _feature_names(state),
        _model_metadata(state)
    )

    path = explainer.extract_path(
        state["sample"]
    )

    return {
        **state,
        "decision_path": path
    }


def feature_importance_node(state):

    explainer = select_feature_importance_explainer(
        state["model"],
        _feature_names(state),
        _model_metadata(state)
    )

    importance = (
        explainer
        .get_top_features(5)
        .to_dict(
            orient="records"
        )
    )

    return {
        **state,
        "feature_importance": importance
    }


def counterfactual_node(state):

    explainer = CounterfactualExplainer(
        state["model"],
        _feature_names(state),
    )

    result = explainer.explain(
        state["sample"],
        target=state.get("counterfactual_target", 1),
    )

    return {
        **state,
        "counterfactual": result,
        "prediction": str(result["baseline_prediction"]),
    }


def local_explanation_node(state):
    """
    Compute SHAP local explanations (feature contributions) for a sample.
    """
    explainer = select_local_explainer(
        state["model"],
        _feature_names(state),
        _model_metadata(state)
    )

    if explainer is None:
        return state

    # Ensure prediction exists in state
    prediction = state.get("prediction")
    if prediction is None:
        model = state["model"]
        prediction = str(model.predict(state["sample"].reshape(1, -1))[0])

    result = explainer.explain_instance(
        state["sample"],
        prediction=prediction
    )

    return {
        **state,
        "local_explanation": result,
        "prediction": prediction
    }


def explanation_node(state):

    prompt = build_explanation_prompt(state)

    provider = build_provider(
        ConfigLoader(state["config_path"])
    )

    return {
        **state,
        "explanation": provider.generate(prompt)
    }