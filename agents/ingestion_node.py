"""
ingestion_node.py

LangGraph entry node that builds the unified ProjectKnowledge object
from config-driven artifact paths and populates the agent state.

Downstream nodes read normalized model, feature and context data from
the ``knowledge`` object rather than loading artifacts themselves.
"""

from ingestion.project_knowledge import ProjectKnowledge
from utils.config_loader import ConfigLoader


def ingestion_node(state):
    """
    Build the project knowledge object and populate agent state.

    Reads the model, data dictionary and project context paths from the
    configuration referenced by ``state["config_path"]``. Any missing
    artifact is handled gracefully by :class:`ProjectKnowledge` and
    surfaced through the validation report.

    Args:
        state: The current :class:`ExplainabilityState`. Must contain a
            ``config_path`` pointing at the project configuration.

    Returns:
        The updated state with ``knowledge`` (the normalized knowledge
        object) and ``model`` (the live loaded model, or ``None``).
    """
    config = ConfigLoader(state["config_path"])

    builder = ProjectKnowledge(
        model_path=config.get("ingestion", "model_path"),
        dictionary_path=config.get("ingestion", "data_dictionary_path"),
        context_path=config.get("ingestion", "project_context_path"),
    )

    knowledge = builder.build()

    return {
        **state,
        "knowledge": knowledge,
        "model": builder.model,
    }
