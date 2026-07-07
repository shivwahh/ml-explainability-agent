from typing import Any, TypedDict


class ExplainabilityState(TypedDict, total=False):
    """
    Shared state for the explainability agent graph.

    ``config_path`` drives ingestion; ``knowledge`` and ``model`` are
    populated by the ingestion node from the unified ProjectKnowledge
    object and consumed by the downstream explanation nodes.
    """

    config_path: str
    question: str
    sample: Any
    knowledge: dict
    model: Any
    intent: str
    prediction: str
    decision_path: dict
    feature_importance: list
    explanation: str