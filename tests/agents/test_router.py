from langgraph.graph import END

from agents.router import (
    router_node,
    route_entry,
    route_after_prediction,
    route_after_decision_path,
    route_after_feature_importance,
)


def _write_config(tmp_path):
    config_path = tmp_path / "project_config.yaml"
    config_path.write_text(
        "routing:\n"
        '  default_intent: "full_explanation"\n'
        "  intents:\n"
        "    decision_path:\n"
        '      - "why did"\n'
        "    feature_importance:\n"
        '      - "which features"\n'
        "    prediction:\n"
        '      - "predict"\n',
        encoding="utf-8",
    )
    return config_path


def test_router_node_sets_intent(tmp_path):
    config_path = _write_config(tmp_path)

    state = {
        "config_path": str(config_path),
        "question": "Why did the model decide this?",
    }
    result = router_node(state)

    assert result["intent"] == "decision_path"


def test_router_node_defaults_to_full(tmp_path):
    config_path = _write_config(tmp_path)

    state = {
        "config_path": str(config_path),
        "question": "Give me a general overview.",
    }
    result = router_node(state)

    assert result["intent"] == "full_explanation"


def test_route_entry_maps_intent_to_first_node():
    assert route_entry({"intent": "decision_path"}) == "decision_path"
    assert route_entry(
        {"intent": "feature_importance"}
    ) == "feature_importance"
    assert route_entry({"intent": "prediction"}) == "prediction"
    assert route_entry({"intent": "full_explanation"}) == "prediction"
    assert route_entry({}) == "prediction"


def test_post_node_routes_continue_only_for_full():
    full = {"intent": "full_explanation"}
    single = {"intent": "prediction"}

    assert route_after_prediction(full) == "continue"
    assert route_after_prediction(single) == "end"

    assert route_after_decision_path(full) == "continue"
    assert route_after_decision_path({"intent": "decision_path"}) == "end"

    assert route_after_feature_importance(full) == "continue"
    assert route_after_feature_importance(
        {"intent": "feature_importance"}
    ) == "end"


def test_graph_compiles_with_router():
    import agents.explainability_graph as graph_module

    nodes = list(graph_module.graph.get_graph().nodes)

    assert "router" in nodes
    assert "ingestion" in nodes
    # END constant is exported and used by the routing maps
    assert END is not None
