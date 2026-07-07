"""
routing

Intent routing tools for the explainability agent.

Provides configuration-driven classification of a user question into a
normalized intent, with no LLM dependency, so the orchestration layer
can route to the relevant explanation branch.
"""

from tools.routing.intent_classifier import IntentClassifier

__all__ = ["IntentClassifier"]
