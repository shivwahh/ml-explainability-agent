from prompts.explanation_prompt import build_explanation_prompt


def _knowledge():
    return {
        "features": [
            {
                "name": "petal length (cm)",
                "meaning": "Length of the petal",
                "unit": "cm",
            },
            {
                "name": "petal width (cm)",
                "meaning": "Width of the petal",
                "unit": "cm",
            },
        ],
        "context": {
            "Project Name": "Iris Species Classifier",
            "Business Objective": "Classify an iris flower.",
        },
    }


def test_prompt_includes_feature_meaning_for_referenced_feature():
    state = {
        "knowledge": _knowledge(),
        "question": "Why this species?",
        "prediction": "0",
        "decision_path": {
            "leaf_node": 1,
            "path": [{"feature": "petal length (cm)", "condition": "<= 2.45"}],
        },
        "feature_importance": [
            {"feature": "petal length (cm)", "importance": 0.6},
        ],
    }

    prompt = build_explanation_prompt(state)

    assert "Feature meanings:" in prompt
    assert "petal length (cm): Length of the petal" in prompt
    assert "unit: cm" in prompt


def test_prompt_includes_project_context():
    state = {
        "knowledge": _knowledge(),
        "question": "Why?",
        "prediction": "0",
        "decision_path": {"path": []},
        "feature_importance": [],
    }

    prompt = build_explanation_prompt(state)

    assert "Business Context:" in prompt
    assert "Iris Species Classifier" in prompt


def test_prompt_omits_sections_when_knowledge_missing():
    state = {
        "question": "Why?",
        "prediction": "0",
        "decision_path": {"path": []},
        "feature_importance": [],
    }

    prompt = build_explanation_prompt(state)

    assert "Feature meanings:" not in prompt
    assert "Business Context:" not in prompt
    assert "Why?" in prompt


def test_prompt_handles_forest_feature_usage():
    state = {
        "knowledge": _knowledge(),
        "question": "Why?",
        "prediction": "0",
        "decision_path": {
            "is_ensemble": True,
            "n_estimators": 5,
            "feature_usage": [{"feature": "petal width (cm)", "tree_count": 4}],
        },
        "feature_importance": [],
    }

    prompt = build_explanation_prompt(state)

    assert "petal width (cm): Width of the petal" in prompt
