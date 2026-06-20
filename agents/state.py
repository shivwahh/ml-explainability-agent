from typing import TypedDict


class ExplainabilityState(
    TypedDict,
    total=False
):
    question: str

    sample

    prediction: str

    decision_path: dict

    feature_importance: list

    explanation: str