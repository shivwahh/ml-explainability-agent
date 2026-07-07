"""
app.py

Streamlit playground for the ML Explainability Agent.

Explore everything built so far from a browser:

* config-driven ingestion into a unified ProjectKnowledge object,
* model framework adapter metadata and the validation report,
* config-driven intent routing for a question, and
* decision-path and feature-importance explanations for a sample.

Run:
    streamlit run ui/app.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent_runner import run_agent  # noqa: E402
from ingestion.project_knowledge import ProjectKnowledge  # noqa: E402
from tools.routing.intent_classifier import IntentClassifier  # noqa: E402
from tools.tree_reader.decision_path_extractor import (  # noqa: E402
    DecisionPathExtractor,
)
from tools.tree_reader.feature_importance_extractor import (  # noqa: E402
    FeatureImportanceExtractor,
)
from utils.config_loader import ConfigLoader  # noqa: E402


DEMO_CONFIG = ROOT / "configs" / "demo_config.yaml"


@st.cache_data(show_spinner=False)
def build_knowledge(config_path: str):
    """Load config and build the knowledge object (cached by path)."""
    config = ConfigLoader(config_path)

    builder = ProjectKnowledge(
        model_path=config.get("ingestion", "model_path"),
        dictionary_path=config.get("ingestion", "data_dictionary_path"),
        context_path=config.get("ingestion", "project_context_path"),
    )
    knowledge = builder.build()

    routing = {
        "intents": config.get("routing", "intents", default={}),
        "default_intent": config.get(
            "routing", "default_intent", default="full_explanation"
        ),
    }

    return knowledge, builder.model, routing


def run_with_provider(config_path, question, sample, provider, model):
    """Run the full agent end to end, overriding the explanation provider.

    Writes a temporary config that keeps the base ingestion/routing
    sections but swaps in the chosen provider/model, so the UI can switch
    between the offline echo provider and a live LLM without editing the
    project config on disk.
    """
    base = ConfigLoader(config_path).as_dict()
    base["explanation"] = {"provider": provider, "model": model}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(base, handle)
        temp_path = handle.name

    try:
        return run_agent(temp_path, question, sample)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def main():
    st.set_page_config(page_title="ML Explainability Playground", layout="wide")
    st.title("ML Explainability Agent — Playground")

    default_config = str(DEMO_CONFIG) if DEMO_CONFIG.exists() else ""
    config_path = st.sidebar.text_input(
        "Config path",
        value=default_config,
        help="Path to a project config YAML (see configs/demo_config.yaml).",
    )

    if not config_path or not Path(config_path).exists():
        st.warning(
            "Enter a valid config path. Generate the demo artifacts with "
            "`python scripts/generate_demo_artifacts.py`."
        )
        return

    knowledge, model, routing = build_knowledge(config_path)
    model_meta = knowledge["model"] or {}
    feature_names = model_meta.get("feature_names", [])

    tab_knowledge, tab_routing, tab_explain, tab_run = st.tabs(
        ["Knowledge", "Routing", "Explain a sample", "End-to-end run"]
    )

    with tab_knowledge:
        st.subheader("Model metadata")
        st.json(model_meta)

        st.subheader("Validation report")
        report = knowledge["validation"]
        st.write("Valid:", report["valid"])
        if report["errors"]:
            st.error("\n".join(report["errors"]))
        if report["warnings"]:
            st.warning("\n".join(report["warnings"]))

        st.subheader("Data dictionary features")
        st.write(knowledge["features"])

        st.subheader("Project context")
        st.write(knowledge["context"])

    with tab_routing:
        st.subheader("Intent classification")
        classifier = IntentClassifier(
            intents=routing["intents"],
            default_intent=routing["default_intent"],
        )
        question = st.text_input(
            "Ask a question",
            value="Why did the model choose this class?",
        )
        if question:
            st.metric("Resolved intent", classifier.classify(question))
        st.caption("Intent keyword map (from config):")
        st.json(routing["intents"])

    with tab_explain:
        if model is None or not feature_names:
            st.info("No fitted model with feature names available in this config.")
            return

        st.subheader("Sample input")
        values = []
        columns = st.columns(min(len(feature_names), 4))
        for index, name in enumerate(feature_names):
            with columns[index % len(columns)]:
                values.append(st.number_input(name, value=0.0, format="%.2f"))

        sample = np.array(values, dtype=float)

        if st.button("Explain"):
            prediction = model.predict(sample.reshape(1, -1))[0]
            st.metric("Prediction", str(prediction))

            st.subheader("Decision path")
            path = DecisionPathExtractor(model, feature_names).extract_path(sample)
            st.write([rule["condition"] for rule in path["path"]])

            st.subheader("Top features")
            importance = FeatureImportanceExtractor(
                model, feature_names
            ).get_top_features(len(feature_names))
            st.dataframe(importance)

    with tab_run:
        st.subheader("Run the full agent end to end")
        st.caption(
            "Ingestion → routing → explanation. The explanation provider "
            "turns the technical result into a business justification."
        )

        if model is None or not feature_names:
            st.info(
                "No fitted model with feature names available in this config."
            )
            return

        run_question = st.text_input(
            "Business question",
            value="Give a business explanation of this decision.",
            key="run_question",
        )

        provider = st.selectbox(
            "Explanation provider",
            options=["echo", "openai"],
            help=(
                "echo runs fully offline and shows the business-context "
                "prompt. openai calls the LLM (needs OPENAI_API_KEY)."
            ),
        )
        model_name = st.text_input(
            "LLM model",
            value="gpt-4.1-mini",
            help="Used only when the provider is openai.",
        )

        st.markdown("**Sample input**")
        run_values = []
        run_columns = st.columns(min(len(feature_names), 4))
        for index, name in enumerate(feature_names):
            with run_columns[index % len(run_columns)]:
                run_values.append(
                    st.number_input(
                        name, value=0.0, format="%.2f", key=f"run_{name}"
                    )
                )

        run_sample = np.array(run_values, dtype=float)

        if st.button("Run agent"):
            with st.spinner("Running the explainability agent..."):
                try:
                    result = run_with_provider(
                        config_path,
                        run_question,
                        run_sample,
                        provider,
                        model_name,
                    )
                except Exception as error:  # surface LLM/config errors
                    st.error(f"Agent run failed: {error}")
                    return

            if result.get("prediction") is not None:
                st.metric("Prediction", str(result["prediction"]))

            st.subheader("Business justification")
            explanation = result.get("explanation")
            if explanation:
                st.markdown(explanation)
            else:
                st.info(
                    "This question routed to a single branch; no full "
                    "explanation was generated. Ask a broader question for "
                    "the business justification."
                )

            with st.expander("Decision path"):
                st.write(result.get("decision_path"))

            with st.expander("Feature importance"):
                st.write(result.get("feature_importance"))


if __name__ == "__main__":
    main()
