"""
_shared.py

Shared helpers and constants for the ML Explainability Playground pages.

The Streamlit entry point (``ui/app.py``) renders the sidebar artifact
selectors and stores the chosen paths in ``st.session_state``. Each page
calls :func:`get_context` to (re)build the cached knowledge object and
derive the maps it needs, so page files stay small and focused on display.
"""

import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent_runner import run_agent  # noqa: E402
from ingestion.project_knowledge import ProjectKnowledge  # noqa: E402


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
        "counterfactual": [
            "what if",
            "make prediction",
            "flip",
            "change",
            "counterfactual",
        ],
        "prediction": [
            "what will",
            "predict",
            "prediction",
            "outcome",
        ],
    },
}


# Latest OpenAI GPT models offered in the "LLM model" dropdown. Used only when
# the explanation provider is "openai". The first entry is the default.
GPT_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
]

# Latest Google Gemini models offered in the "LLM model" dropdown. Used only when
# the explanation provider is "gemini". The first entry is the default.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


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


def get_context():
    """Build the shared page context from the selected sidebar artifacts.

    Reads the artifact paths stashed in ``st.session_state`` by the entry
    point, (re)builds the cached knowledge object and derives the maps each
    page needs. If no valid model is selected, it renders a warning and
    halts the page via ``st.stop``.

    Returns:
        A ``SimpleNamespace`` with the model, knowledge, routing map and the
        derived feature helper/default maps.
    """
    paths = st.session_state.get("artifact_paths") or {}
    model_path = paths.get("model_path")

    if not model_path or not Path(model_path).exists():
        st.warning(
            "Select a valid model artifact from the sidebar. Generate demo "
            "or test models with the notebooks in `notebooks/` or "
            "`python scripts/generate_demo_artifacts.py`.",
            icon=":material/warning:",
        )
        st.stop()

    knowledge, model = build_knowledge(
        model_path, paths.get("dictionary_path"), paths.get("context_path")
    )
    model_meta = knowledge["model"] or {}
    feature_names = model_meta.get("feature_names", [])

    return SimpleNamespace(
        model_path=model_path,
        dictionary_path=paths.get("dictionary_path"),
        context_path=paths.get("context_path"),
        routing=DEFAULT_ROUTING,
        knowledge=knowledge,
        model=model,
        model_meta=model_meta,
        feature_names=feature_names,
        help_by_name=feature_help_map(knowledge["features"]),
        default_by_name=default_value_map(knowledge["features"]),
        hint=_dataset_hint(model_path),
    )
