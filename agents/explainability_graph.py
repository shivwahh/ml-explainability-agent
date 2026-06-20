from langgraph.graph import (
    StateGraph,
    END
)

from agents.state import (
    ExplainabilityState
)

from agents.nodes import (
    planner_node,
    prediction_node,
    decision_path_node,
    feature_importance_node,
    explanation_node
)

builder = StateGraph(
    ExplainabilityState
)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "prediction",
    prediction_node
)

builder.add_node(
    "decision_path",
    decision_path_node
)

builder.add_node(
    "feature_importance",
    feature_importance_node
)

builder.add_node(
    "explanation",
    explanation_node
)

builder.set_entry_point(
    "planner"
)

builder.add_edge(
    "planner",
    "prediction"
)

builder.add_edge(
    "prediction",
    "decision_path"
)

builder.add_edge(
    "decision_path",
    "feature_importance"
)

builder.add_edge(
    "feature_importance",
    "explanation"
)

builder.add_edge(
    "explanation",
    END
)

graph = builder.compile()
