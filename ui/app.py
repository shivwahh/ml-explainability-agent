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

import re
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
from tools.explainers.explainer_selector import (  # noqa: E402
    select_decision_path_explainer,
    select_feature_importance_explainer,
)


# Default intent-routing map used when no config file drives the UI. Mirrors the
# routing section of configs/demo_config.yaml so manual artifact selection still
# routes questions to the right explanation branch.
DEFAULT_ROUTING = {
    "default_intent": "full_explanation",
    "intents": {
        "decision_path": [
            "decision path",
            "why did",
            "how did",
            "which rules",
            "path",
        ],
        "feature_importance": [
            "feature importance",
            "most important",
            "which features",
            "influential",
            "drivers",
        ],
        "prediction": [
            "what will",
            "predict",
            "prediction",
            "outcome",
        ],
    },
}


def discover(patterns):
    """Return existing files under the project root matching any glob pattern.

    Results are de-duplicated while preserving first-seen order so a file
    matched by several patterns appears once.
    """
    found = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def _dataset_hint(model_path):
    """Guess the dataset folder key from a model filename.

    e.g. 'adult_income_xgboost.joblib' -> 'adult_income'. Falls back to the
    first underscore-separated token of the stem so unknown models still get
    a best-effort match against their data folder.
    """
    if not model_path:
        return ""
    stem = Path(model_path).stem
    known = ("adult_income", "california_housing", "breast_cancer", "demo")
    for key in known:
        if key in stem:
            return key
    return stem.split("_")[0]


def artifact_selector(label, patterns, *, required, prefer="", key_suffix=""):
    """Render a sidebar selector for an artifact and return the chosen path.

    Lists discovered files as options plus a "Custom path…" escape hatch.
    Optional artifacts also get a "(none)" choice. When ``prefer`` is given,
    the first option whose path contains it is selected by default so the
    dictionary/context can auto-match the chosen model. Returns an absolute
    path string, or ``None`` when nothing is selected.
    """
    files = discover(patterns)
    options = [path.relative_to(ROOT).as_posix() for path in files]

    # Real files first so an optional selector defaults to a file, not "(none)".
    choices = options + ([] if required else ["(none)"]) + ["Custom path…"]

    default_index = 0
    if prefer:
        for index, option in enumerate(choices):
            if option not in ("(none)", "Custom path…") and prefer in option:
                default_index = index
                break

    choice = st.sidebar.selectbox(
        label, choices, index=default_index, key=f"sel_{label}{key_suffix}"
    )

    if choice == "Custom path…":
        custom = st.sidebar.text_input(
            f"{label} — custom path", key=f"custom_{label}"
        )
        return custom or None
    if choice == "(none)":
        return None

    return str((ROOT / choice).resolve())


@st.cache_data(show_spinner=False)
def build_knowledge(model_path, dictionary_path, context_path):
    """Build the knowledge object from explicit artifact paths (cached)."""
    builder = ProjectKnowledge(
        model_path=model_path,
        dictionary_path=dictionary_path,
        context_path=context_path,
    )
    knowledge = builder.build()

    return knowledge, builder.model


def run_with_provider(
    model_path,
    dictionary_path,
    context_path,
    routing,
    question,
    sample,
    provider,
    model,
):
    """Run the full agent end to end from explicit artifact paths.

    Writes a temporary config assembled from the selected artifacts, the
    default routing map and the chosen explanation provider/model, so the
    UI can run the agent without a pre-existing config file on disk.
    """
    base = {
        "ingestion": {
            "model_path": model_path,
            "data_dictionary_path": dictionary_path,
            "project_context_path": context_path,
        },
        "routing": routing,
        "explanation": {"provider": provider, "model": model},
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(base, handle)
        temp_path = handle.name

    try:
        return run_agent(temp_path, question, sample)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def feature_help_map(feature_records):
    """Map feature name -> help string (meaning + allowed range) for inputs."""
    help_by_name = {}
    for record in feature_records or []:
        name = record.get("name")
        if not name:
            continue
        parts = []
        if record.get("meaning"):
            parts.append(str(record["meaning"]))
        if record.get("allowed_range"):
            parts.append(f"Allowed: {record['allowed_range']}")
        help_by_name[str(name)] = " | ".join(parts) or None
    return help_by_name


def _range_midpoint(allowed_range):
    """Return the midpoint of an 'Allowed Range' string, or None.

    Handles plain 'min-max' ranges (including negatives and decimals) and
    ignores any trailing parenthetical, e.g. '0-8 (0=Private; ...)'. A single
    value is returned as-is.
    """
    if not allowed_range:
        return None
    text = str(allowed_range).split("(")[0]
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    if len(numbers) == 1:
        return round(float(numbers[0]), 4)
    return round((float(numbers[0]) + float(numbers[-1])) / 2, 4)


def default_value_map(feature_records):
    """Map feature name -> a representative default value for sample inputs.

    Prefers an explicit median/sample column from the data dictionary
    (any of 'median', 'sample_value', 'example', 'default'); otherwise
    falls back to the midpoint of the 'Allowed Range'. Missing values
    default to 0.0 at the call site.
    """
    preferred_keys = ("median", "sample_value", "example", "default")
    defaults = {}
    for record in feature_records or []:
        name = record.get("name")
        if not name:
            continue
        value = None
        for key in preferred_keys:
            if record.get(key) is not None:
                try:
                    value = round(float(record[key]), 4)
                except (TypeError, ValueError):
                    value = None
                break
        if value is None:
            value = _range_midpoint(record.get("allowed_range"))
        if value is not None:
            defaults[str(name)] = value
    return defaults


def main():
    st.set_page_config(page_title="ML Explainability Playground", layout="wide")
    st.title("ML Explainability Agent — Playground")

    st.sidebar.header("Artifacts")
    st.sidebar.caption(
        "Pick any model, data dictionary and project context, then enter a "
        "sample below to run the model."
    )

    model_path = artifact_selector(
        "Model",
        ["models/*.joblib", "models/*.pkl"],
        required=True,
    )

    # Auto-match the data dictionary and context to the selected model so the
    # sample inputs pre-fill from the right medians. The hint is folded into the
    # widget key so changing the model re-selects the matching artifacts.
    hint = _dataset_hint(model_path)

    dictionary_path = artifact_selector(
        "Data dictionary",
        ["data/**/data_dictionary.csv", "data/**/*.xlsx"],
        required=False,
        prefer=hint,
        key_suffix=f"_{hint}",
    )
    context_path = artifact_selector(
        "Project context",
        ["data/**/project_context.md", "data/**/*.txt"],
        required=False,
        prefer=hint,
        key_suffix=f"_{hint}",
    )

    if not model_path or not Path(model_path).exists():
        st.warning(
            "Select a valid model artifact from the sidebar. Generate demo "
            "or test models with the notebooks in `notebooks/` or "
            "`python scripts/generate_demo_artifacts.py`."
        )
        return

    knowledge, model = build_knowledge(model_path, dictionary_path, context_path)
    routing = DEFAULT_ROUTING
    model_meta = knowledge["model"] or {}
    feature_names = model_meta.get("feature_names", [])
    help_by_name = feature_help_map(knowledge["features"])
    default_by_name = default_value_map(knowledge["features"])

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
        st.caption("Intent keyword map (default routing):")
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
                values.append(
                    st.number_input(
                        name,
                        value=float(default_by_name.get(name, 0.0)),
                        format="%.2f",
                        help=help_by_name.get(name),
                    )
                )

        sample = np.array(values, dtype=float)

        if st.button("Explain"):
            prediction = model.predict(sample.reshape(1, -1))[0]
            st.metric("Prediction", str(prediction))

            st.subheader("Decision path")
            try:
                path_explainer = select_decision_path_explainer(
                    model, feature_names, model_meta
                )
                path = path_explainer.extract_path(sample)
                st.write([rule["condition"] for rule in path["path"]])
            except Exception as error:  # e.g. path not supported for this model
                st.info(f"Decision path unavailable for this model: {error}")

            st.subheader("Top features")
            importance = select_feature_importance_explainer(
                model, feature_names, model_meta
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
                        name,
                        value=float(default_by_name.get(name, 0.0)),
                        format="%.2f",
                        key=f"run_{hint}_{name}",
                        help=help_by_name.get(name),
                    )
                )

        run_sample = np.array(run_values, dtype=float)

        if st.button("Run agent"):
            with st.spinner("Running the explainability agent..."):
                try:
                    result = run_with_provider(
                        model_path,
                        dictionary_path,
                        context_path,
                        routing,
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
