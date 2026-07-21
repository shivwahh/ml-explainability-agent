"""
app.py

Streamlit playground for the ML Explainability Agent (multi-page entry point).

Renders the shared sidebar artifact selectors and the top navigation, then
dispatches to one of the focused pages in ``app_pages/``:

* Knowledge   — model metadata, validation and the data dictionary,
* Routing     — config-driven intent classification for a question,
* Explain     — decision-path and feature-importance for one sample, and
* Run         — the full agent end to end with a business justification.

Run:
    streamlit run ui/app.py
"""

import streamlit as st

from _shared import DEFAULT_ROUTING, _dataset_hint, artifact_selector

st.set_page_config(
    page_title="ML Explainability Playground",
    page_icon=":material/network_intelligence:",
    layout="wide",
)

st.sidebar.header(":material/folder: Artifacts")
st.sidebar.caption(
    "Pick any model, data dictionary and project context, then explore the "
    "pages above."
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

# Shared selection consumed by every page via _shared.get_context().
st.session_state["artifact_paths"] = {
    "model_path": model_path,
    "dictionary_path": dictionary_path,
    "context_path": context_path,
}
st.session_state["routing"] = DEFAULT_ROUTING

pages = st.navigation(
    [
        st.Page(
            "app_pages/knowledge.py",
            title="Knowledge",
            icon=":material/database:",
        ),
        st.Page(
            "app_pages/routing.py",
            title="Routing",
            icon=":material/route:",
        ),
        st.Page(
            "app_pages/explain.py",
            title="Explain a sample",
            icon=":material/lightbulb:",
        ),
        st.Page(
            "app_pages/run.py",
            title="End-to-end run",
            icon=":material/play_arrow:",
        ),
    ],
    position="top",
)

st.title(":material/network_intelligence: ML explainability agent")
st.caption(
    "Explore config-driven ingestion, intent routing and model explanations "
    "from your browser."
)

pages.run()
