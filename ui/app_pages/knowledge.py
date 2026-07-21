"""Knowledge page — model metadata, validation and the data dictionary."""

import streamlit as st

from _shared import get_context

ctx = get_context()
meta = ctx.model_meta

st.subheader("Model overview")
cols = st.columns(4)
cols[0].metric("Model type", str(meta.get("model_type", "—")))
cols[1].metric("Task", str(meta.get("task_type", "—")))
cols[2].metric(
    "Features", str(meta.get("n_features", len(ctx.feature_names)) or "—")
)
cols[3].metric("Framework", str(meta.get("framework", "—")))

tree = meta.get("tree_details") or {}
if tree.get("is_ensemble"):
    st.caption(
        f"Ensemble model with {tree.get('n_estimators', '?')} estimators."
    )

classes = meta.get("classes")
if classes is not None:
    st.caption(f"Classes: {', '.join(str(c) for c in classes)}")

# Validation report as badges + messages instead of a raw dict.
report = ctx.knowledge["validation"]
with st.container(border=True):
    st.subheader("Validation")
    if report["valid"]:
        st.badge("Valid", icon=":material/check_circle:", color="green")
    else:
        st.badge("Invalid", icon=":material/error:", color="red")
    if report["errors"]:
        st.error("\n".join(report["errors"]), icon=":material/error:")
    if report["warnings"]:
        st.warning("\n".join(report["warnings"]), icon=":material/warning:")
    if not report["errors"] and not report["warnings"]:
        st.caption("No errors or warnings.")

# Data dictionary as a clean table instead of a raw list of dicts.
st.subheader("Feature dictionary")
features = ctx.knowledge["features"]
if features:
    st.dataframe(
        features,
        hide_index=True,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Feature"),
            "meaning": st.column_config.TextColumn("Meaning", width="large"),
            "data_type": st.column_config.TextColumn("Type"),
            "unit": st.column_config.TextColumn("Unit"),
            "allowed_range": st.column_config.TextColumn(
                "Allowed range", width="medium"
            ),
            "median": st.column_config.NumberColumn("Median"),
        },
    )
else:
    st.caption("No data dictionary provided for this model.")

# Project context rendered as readable sections instead of a raw dict.
st.subheader("Project context")
context = ctx.knowledge["context"]
if context:
    for key, value in context.items():
        with st.container(border=True):
            st.markdown(f"**{key}**")
            st.markdown(str(value))
else:
    st.caption("No project context provided for this model.")

# Power-user details kept out of the way in collapsed expanders.
hyperparameters = meta.get("hyperparameters") or {}
if hyperparameters:
    with st.expander("Hyperparameters", icon=":material/tune:"):
        st.dataframe(
            [
                {"parameter": key, "value": str(value)}
                for key, value in hyperparameters.items()
            ],
            hide_index=True,
            width="stretch",
        )

with st.expander("Raw model metadata (JSON)", icon=":material/data_object:"):
    st.json(meta)
