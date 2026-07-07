"""
router.py

Intent-based routing for the explainability agent graph.

The router node classifies the user question into a normalized intent
using the config-driven :class:`IntentClassifier`, and the routing
functions drive the conditional edges that dispatch to the relevant
explanation branch. An unknown or general question falls back to the
full explanation pipeline.
"""

from langgraph.graph import END

from tools.routing.intent_classifier import IntentClassifier
from utils.config_loader import ConfigLoader


INTENT_PREDICTION = "prediction"
INTENT_DECISION_PATH = "decision_path"
INTENT_FEATURE_IMPORTANCE = "feature_importance"
INTENT_FULL = "full_explanation"


def router_node(state):
    """
    Classify the question and record the resolved intent on the state.

    Args:
        state: The current state; must contain ``config_path`` and may
            contain ``question``.

    Returns:
        The updated state with the resolved ``intent``.
    """
    config = ConfigLoader(state["config_path"])

    classifier = IntentClassifier(
        intents=config.get("routing", "intents", default={}),
        default_intent=config.get(
            "routing", "default_intent", default=INTENT_FULL
        ),
    )

    intent = classifier.classify(state.get("question", ""))

    return {**state, "intent": intent}


def route_entry(state) -> str:
    """
    Select the first node to run based on the resolved intent.

    Both the single ``prediction`` intent and the full pipeline enter at
    the prediction node; the post-node routers decide whether to stop or
    continue.
    """
    intent = state.get("intent") or INTENT_FULL

    if intent == INTENT_DECISION_PATH:
        return INTENT_DECISION_PATH

    if intent == INTENT_FEATURE_IMPORTANCE:
        return INTENT_FEATURE_IMPORTANCE

    return INTENT_PREDICTION


def route_after_prediction(state) -> str:
    """
    Continue to the decision path only when the full pipeline is active.
    """
    return "continue" if state.get("intent") == INTENT_FULL else "end"


def route_after_decision_path(state) -> str:
    """
    Continue to feature importance only for the full pipeline.
    """
    return "continue" if state.get("intent") == INTENT_FULL else "end"


def route_after_feature_importance(state) -> str:
    """
    Continue to the explanation node only for the full pipeline.
    """
    return "continue" if state.get("intent") == INTENT_FULL else "end"


ENTRY_ROUTES = {
    INTENT_PREDICTION: INTENT_PREDICTION,
    INTENT_DECISION_PATH: INTENT_DECISION_PATH,
    INTENT_FEATURE_IMPORTANCE: INTENT_FEATURE_IMPORTANCE,
}

PREDICTION_ROUTES = {"continue": INTENT_DECISION_PATH, "end": END}

DECISION_PATH_ROUTES = {"continue": INTENT_FEATURE_IMPORTANCE, "end": END}

FEATURE_IMPORTANCE_ROUTES = {"continue": "explanation", "end": END}
