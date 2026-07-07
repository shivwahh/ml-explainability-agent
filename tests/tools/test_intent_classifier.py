from tools.routing.intent_classifier import IntentClassifier


def _intents():
    return {
        "decision_path": ["decision path", "why did", "path"],
        "feature_importance": ["most important", "which features"],
        "prediction": ["predict", "what will"],
    }


def _classifier():
    return IntentClassifier(
        intents=_intents(),
        default_intent="full_explanation",
    )


def test_classifies_prediction():
    assert _classifier().classify(
        "What will the model predict for this customer?"
    ) == "prediction"


def test_classifies_decision_path():
    assert _classifier().classify(
        "Why did the model reach this decision?"
    ) == "decision_path"


def test_classifies_feature_importance():
    assert _classifier().classify(
        "Which features are most important?"
    ) == "feature_importance"


def test_unknown_question_falls_back_to_default():
    assert _classifier().classify(
        "Tell me about the weather."
    ) == "full_explanation"


def test_empty_question_falls_back_to_default():
    assert _classifier().classify("") == "full_explanation"


def test_first_matching_intent_wins():
    classifier = IntentClassifier(
        intents={
            "decision_path": ["path"],
            "prediction": ["predict"],
        },
        default_intent="full_explanation",
    )

    # "path" (decision_path) is checked before "predict" (prediction)
    assert classifier.classify(
        "Show the path and predict the result"
    ) == "decision_path"
