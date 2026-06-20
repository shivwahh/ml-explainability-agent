from openai import OpenAI

from tools.tree_reader.decision_path_extractor import (
    DecisionPathExtractor
)

from tools.tree_reader.feature_importance_extractor import (
    FeatureImportanceExtractor
)

client = OpenAI()


def planner_node(state):

    print("Planner Node")

    return state

def prediction_node(
    state,
    model,
    target_names
):

    prediction = model.predict(
        state["sample"].reshape(1, -1)
    )[0]

    return {
        **state,
        "prediction": target_names[
            prediction
        ]
    }
    



def decision_path_node(
    state,
    model,
    feature_names
):

    extractor = (
        DecisionPathExtractor(
            model,
            feature_names
        )
    )

    path = (
        extractor.extract_path(
            state["sample"]
        )
    )

    return {
        **state,
        "decision_path": path
    }
    



def feature_importance_node(
    state,
    model,
    feature_names
):

    extractor = (
        FeatureImportanceExtractor(
            model,
            feature_names
        )
    )

    importance = (
        extractor
        .get_top_features(5)
        .to_dict(
            orient="records"
        )
    )

    return {
        **state,
        "feature_importance": importance
    }
    
def explanation_node(state):

    prompt = f"""
You are an Explainable AI expert.

Question:
{state['question']}

Prediction:
{state['prediction']}

Decision Path:
{state['decision_path']}

Top Features:
{state['feature_importance']}

Explain in business-friendly language.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        **state,
        "explanation": (
            response.output_text
        )
    }