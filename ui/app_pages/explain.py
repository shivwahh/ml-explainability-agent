"""Explain page — decision path and feature importance for one sample."""

import numpy as np
import streamlit as st

from _shared import get_context
from tools.explainers.explainer_selector import (
    select_decision_path_explainer,
    select_feature_importance_explainer,
)

ctx = get_context()

if ctx.model is None or not ctx.feature_names:
    st.info(
        "No fitted model with feature names available in this config.",
        icon=":material/info:",
    )
    st.stop()

st.subheader("Sample input")
values = []
columns = st.columns(min(len(ctx.feature_names), 4))
for index, name in enumerate(ctx.feature_names):
    with columns[index % len(columns)]:
        values.append(
            st.number_input(
                name,
                value=float(ctx.default_by_name.get(name, 0.0)),
                format="%.2f",
                key=f"explain_{ctx.hint}_{name}",
                help=ctx.help_by_name.get(name),
            )
        )

sample = np.array(values, dtype=float)

if st.button("Explain", type="primary", icon=":material/lightbulb:"):
    try:
        prediction = ctx.model.predict(sample.reshape(1, -1))[0]
        st.session_state[f"explain_result_{ctx.hint}"] = {
            "prediction": str(prediction),
            "sample": sample.tolist(),
        }
    except Exception as error:  # surface model/predict errors
        st.error(f"Prediction failed: {error}", icon=":material/error:")
        st.session_state.pop(f"explain_result_{ctx.hint}", None)

# Render from session state so results survive reruns (e.g. expander toggles).
stored = st.session_state.get(f"explain_result_{ctx.hint}")
if stored is not None:
    explain_sample = np.array(stored["sample"], dtype=float)

    with st.container(border=True):
        st.metric("Prediction", stored["prediction"])

    st.subheader("Decision path")
    try:
        path_explainer = select_decision_path_explainer(
            ctx.model, ctx.feature_names, ctx.model_meta
        )
        path = path_explainer.extract_path(explain_sample)
        conditions = [rule["condition"] for rule in path["path"]]
        if conditions:
            for condition in conditions:
                st.markdown(f"- {condition}")
        else:
            st.caption("No branching conditions for this sample.")
    except Exception as error:  # path not supported for this model
        st.info(
            f"Decision path unavailable for this model: {error}",
            icon=":material/info:",
        )

    st.subheader("Top features")
    importance = select_feature_importance_explainer(
        ctx.model, ctx.feature_names, ctx.model_meta
    ).get_top_features(len(ctx.feature_names))
    st.dataframe(importance, width="stretch", hide_index=True)

    st.subheader("Local Feature Contributions (SHAP)")
    try:
        from tools.explainers.explainer_selector import select_local_explainer
        local_explainer = select_local_explainer(
            ctx.model, ctx.feature_names, ctx.model_meta
        )
        if local_explainer is not None:
            shap_result = local_explainer.explain_instance(
                explain_sample,
                prediction=stored["prediction"]
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Base Value", f"{shap_result['base_value']:.4f}")
            with col2:
                st.metric("Predicted Value (Margin/Prob)", f"{shap_result['prediction_value']:.4f}")

            import pandas as pd
            df_contrib = pd.DataFrame(shap_result["contributions"])
            st.dataframe(df_contrib, width="stretch", hide_index=True)
        else:
            st.info("SHAP is not available or model is not supported for SHAP.", icon=":material/info:")
    except Exception as error:
        st.info(
            f"SHAP explanation unavailable: {error}",
            icon=":material/info:",
        )

