"""
explanation_prompt.py

Business-context-aware explanation prompt builder.

Composes the technical explanation signals (question, prediction,
decision path, feature importance) with the semantic layer from the
knowledge object - data dictionary feature meanings/units and project
context sections - so the LLM can translate outputs into business
language grounded in the project's own documentation.

Pure functionality: no model or LLM calls.
"""


def _feature_meanings(knowledge: dict) -> dict:
    """
    Build a map of feature name to a human-readable meaning string.

    Combines the data dictionary ``meaning`` and ``unit`` fields when
    present.
    """
    meanings = {}

    for record in knowledge.get("features", []) or []:
        name = record.get("name")

        if not name:
            continue

        meaning = record.get("meaning")
        unit = record.get("unit")

        parts = []

        if meaning:
            parts.append(str(meaning))

        if unit:
            parts.append(f"unit: {unit}")

        meanings[str(name)] = " (".join(parts) + ")" if unit else (
            meaning or ""
        )

    return meanings


def _referenced_features(state: dict) -> list:
    """
    Return the feature names referenced by the explanation signals.

    Handles both a single-tree decision path (``path`` rules) and a
    forest path (``feature_usage`` entries), plus feature importance.
    """
    names = []

    decision_path = state.get("decision_path") or {}

    for rule in decision_path.get("path", []) or []:
        feature = rule.get("feature")
        if feature:
            names.append(feature)

    for entry in decision_path.get("feature_usage", []) or []:
        feature = entry.get("feature")
        if feature:
            names.append(feature)

    for record in state.get("feature_importance", []) or []:
        feature = record.get("feature")
        if feature:
            names.append(feature)

    # Preserve order while de-duplicating.
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    return ordered


def _render_feature_glossary(state: dict, meanings: dict) -> str:
    """
    Render meanings for the features referenced in the explanation.
    """
    lines = []

    for name in _referenced_features(state):
        meaning = meanings.get(name)
        if meaning:
            lines.append(f"- {name}: {meaning}")

    if not lines:
        return ""

    return "Feature meanings:\n" + "\n".join(lines)


def _render_context(knowledge: dict) -> str:
    """
    Render project-context sections when available.
    """
    context = knowledge.get("context") or {}

    if not context:
        return ""

    lines = []

    for heading, body in context.items():
        if not body:
            continue
        lines.append(f"## {heading}\n{body}")

    if not lines:
        return ""

    return "Business Context:\n" + "\n\n".join(lines)


def build_explanation_prompt(state: dict) -> str:
    """
    Build the business-context-aware explanation prompt.

    Args:
        state: The agent state; may contain ``knowledge``, ``question``,
            ``prediction``, ``decision_path`` and ``feature_importance``.

    Returns:
        The rendered prompt string. Sections without data are omitted
        gracefully.
    """
    knowledge = state.get("knowledge", {}) or {}
    meanings = _feature_meanings(knowledge)

    sections = [
        "You are an Explainable AI expert. Translate the model's "
        "technical output into clear business language.",
        f"Question:\n{state.get('question')}",
        f"Prediction:\n{state.get('prediction')}",
    ]

    glossary = _render_feature_glossary(state, meanings)
    if glossary:
        sections.append(glossary)

    sections.append(f"Decision Path:\n{state.get('decision_path')}")
    sections.append(f"Top Features:\n{state.get('feature_importance')}")

    context = _render_context(knowledge)
    if context:
        sections.append(context)

    sections.append(
        "Explain in business-friendly language, referring to features "
        "by their business meaning where available."
    )

    return "\n\n".join(sections)
