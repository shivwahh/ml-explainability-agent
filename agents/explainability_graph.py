from langgraph.graph import (
    StateGraph,
    END
)

from agents.state import (
    ExplainabilityState
)

from agents.ingestion_node import (
    ingestion_node
)

from agents.router import (
    router_node,
    route_entry,
    route_after_prediction,
    route_after_decision_path,
    route_after_feature_importance,
    route_after_local_explanation,
    ENTRY_ROUTES,
    PREDICTION_ROUTES,
    DECISION_PATH_ROUTES,
    FEATURE_IMPORTANCE_ROUTES,
    LOCAL_EXPLANATION_ROUTES
)

from agents.nodes import (
    planner_node,
    prediction_node,
    decision_path_node,
    feature_importance_node,
    local_explanation_node,
    counterfactual_node,
    explanation_node
)

builder = StateGraph(
    ExplainabilityState
)

builder.add_node(
    "ingestion",
    ingestion_node
)

builder.add_node(
    "planner",
    planner_node
)

builder.add_node(
    "router",
    router_node
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
    "counterfactual",
    counterfactual_node
)

builder.add_node(
    "local_explanation",
    local_explanation_node
)

builder.add_node(
    "explanation",
    explanation_node
)

builder.set_entry_point(
    "ingestion"
)

builder.add_edge(
    "ingestion",
    "planner"
)

builder.add_edge(
    "planner",
    "router"
)

builder.add_conditional_edges(
    "router",
    route_entry,
    ENTRY_ROUTES
)

builder.add_conditional_edges(
    "prediction",
    route_after_prediction,
    PREDICTION_ROUTES
)

builder.add_conditional_edges(
    "decision_path",
    route_after_decision_path,
    DECISION_PATH_ROUTES
)

builder.add_conditional_edges(
    "feature_importance",
    route_after_feature_importance,
    FEATURE_IMPORTANCE_ROUTES
)

builder.add_conditional_edges(
    "local_explanation",
    route_after_local_explanation,
    LOCAL_EXPLANATION_ROUTES
)

builder.add_edge(
    "counterfactual",
    "explanation"
)

builder.add_edge(
    "explanation",
    END
)

graph = builder.compile()
