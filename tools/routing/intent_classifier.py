"""
intent_classifier.py

Configuration-driven intent classification.

Maps a natural-language question to a normalized intent using a keyword
map supplied by configuration. This is pure functionality with no LLM
call: the orchestration layer owns any model access.
"""


class IntentClassifier:
    """
    Classify a question into a normalized intent from a keyword map.

    Intents are checked in the order they appear in the supplied mapping;
    the first intent with a matching keyword wins. When nothing matches,
    the configured default intent is returned.
    """

    def __init__(self, intents: dict = None, default_intent: str = "full_explanation"):
        """
        Args:
            intents: Mapping of intent name to an iterable of keyword
                strings. Matching is case-insensitive substring matching.
            default_intent: Intent returned when no keyword matches.
        """
        self.intents = intents or {}
        self.default_intent = default_intent

    def classify(self, question: str) -> str:
        """
        Return the normalized intent for a question.

        Args:
            question: The user's natural-language question.

        Returns:
            The matched intent name, or the default intent when no
            keyword matches.
        """
        text = (question or "").lower()

        for intent, keywords in self.intents.items():
            for keyword in keywords or []:
                if str(keyword).lower() in text:
                    return intent

        return self.default_intent
