"""
agent_runner.py

End-to-end runner for the explainability agent graph.

Assembles the initial ExplainabilityState from a config path plus the
user's question and sample, then invokes the compiled LangGraph agent.
"""

import numpy as np

from agents.explainability_graph import graph


def build_initial_state(config_path, question, sample=None) -> dict:
    """
    Assemble the initial agent state from config and inputs.

    Args:
        config_path: Path to the project configuration.
        question: The user's natural-language question.
        sample: Optional feature values (any array-like) for the record
            being explained; converted to a float numpy array.

    Returns:
        The initial :class:`ExplainabilityState` dictionary.
    """
    state = {
        "config_path": str(config_path),
        "question": question,
    }

    if sample is not None:
        state["sample"] = np.asarray(sample, dtype=float)

    return state


def run_agent(config_path, question, sample=None) -> dict:
    """
    Run the compiled explainability graph end to end.

    Args:
        config_path: Path to the project configuration.
        question: The user's natural-language question.
        sample: Optional feature values for the record being explained.

    Returns:
        The final agent state after the graph completes.
    """
    return graph.invoke(
        build_initial_state(config_path, question, sample)
    )
