"""End-to-end run page — full agent with a business justification."""

import numpy as np
import streamlit as st

from _shared import GPT_MODELS, GEMINI_MODELS, get_context, run_with_provider

ctx = get_context()

st.subheader("Run the full agent end to end")
st.caption(
    "Ingestion → routing → explanation. Predict first, then ask why the model "
    "produced that output to get a business justification."
)

if ctx.model is None or not ctx.feature_names:
    st.info(
        "No fitted model with feature names available in this config.",
        icon=":material/info:",
    )
    st.stop()

provider = st.segmented_control(
    "Explanation provider",
    options=["echo", "openai", "gemini"],
    default="echo",
    help=(
        "echo runs fully offline and shows the business-context prompt. "
        "openai calls OpenAI (needs OPENAI_API_KEY). "
        "gemini calls Google Gemini (needs GEMINI_API_KEY)."
    ),
)

if provider == "gemini":
    models_list = GEMINI_MODELS
    help_text = "Used only when the provider is gemini."
elif provider == "openai":
    models_list = GPT_MODELS
    help_text = "Used only when the provider is openai."
else:
    models_list = ["(none)"]
    help_text = "Not used for echo provider."

model_name = st.selectbox(
    "LLM model",
    options=models_list,
    index=0,
    help=help_text,
    disabled=(provider == "echo" or provider is None),
)

st.markdown("**Sample input**")
run_values = []
run_columns = st.columns(min(len(ctx.feature_names), 4))
for index, name in enumerate(ctx.feature_names):
    with run_columns[index % len(run_columns)]:
        run_values.append(
            st.number_input(
                name,
                value=float(ctx.default_by_name.get(name, 0.0)),
                format="%.2f",
                key=f"run_{ctx.hint}_{name}",
                help=ctx.help_by_name.get(name),
            )
        )

run_sample = np.array(run_values, dtype=float)

# Phase 1: predict first. The prediction (and the sample it came from) is
# stashed in session state so the business-question step below survives the
# rerun that Streamlit triggers on widget input.
if st.button("Run prediction", type="primary", icon=":material/play_arrow:"):
    try:
        prediction = ctx.model.predict(run_sample.reshape(1, -1))[0]
        st.session_state[f"run_prediction_{ctx.hint}"] = str(prediction)
        st.session_state[f"run_sample_{ctx.hint}"] = run_sample.tolist()
        # New prediction invalidates any prior justification.
        st.session_state.pop(f"run_result_{ctx.hint}", None)
    except Exception as error:  # surface model/predict errors
        st.error(f"Prediction failed: {error}", icon=":material/error:")
        st.session_state.pop(f"run_prediction_{ctx.hint}", None)
        st.session_state.pop(f"run_sample_{ctx.hint}", None)
        st.session_state.pop(f"run_result_{ctx.hint}", None)

# Phase 2: once a prediction exists, show it and let the user ask a business
# question about why that output came, then explain it.
if st.session_state.get(f"run_prediction_{ctx.hint}") is not None:
    with st.container(border=True):
        st.metric("Prediction", st.session_state[f"run_prediction_{ctx.hint}"])

    st.markdown(
        "**Ask about this prediction** — e.g. why this output, what drove it, "
        "or what would change it."
    )
    run_question = st.text_input(
        "Business question",
        value="Why did the model produce this prediction?",
        key=f"run_question_{ctx.hint}",
    )

    explain_sample = np.array(
        st.session_state[f"run_sample_{ctx.hint}"], dtype=float
    )

    if st.button(
        "Explain prediction", type="primary", icon=":material/lightbulb:"
    ):
        # Force the full pipeline so any business question (even "why did..."
        # which would otherwise route to a single branch) always yields a
        # business justification.
        explain_routing = {"default_intent": "full_explanation", "intents": {}}
        with st.spinner("Running the explainability agent..."):
            try:
                # Persist the result so the justification and its expanders
                # survive reruns (e.g. opening an expander) instead of
                # vanishing on the next click.
                st.session_state[f"run_result_{ctx.hint}"] = run_with_provider(
                    ctx.model_path,
                    ctx.dictionary_path,
                    ctx.context_path,
                    explain_routing,
                    run_question,
                    explain_sample,
                    provider or "echo",
                    model_name,
                )
            except Exception as error:  # surface LLM/config errors
                st.error(f"Agent run failed: {error}", icon=":material/error:")
                st.session_state.pop(f"run_result_{ctx.hint}", None)

    result = st.session_state.get(f"run_result_{ctx.hint}")
    if result is not None:
        st.subheader("Business justification")
        explanation = result.get("explanation")
        if explanation:
            st.markdown(explanation)
        else:
            st.info(
                "No business justification was generated for this question. "
                "Try rephrasing it, or switch the provider to openai for a "
                "richer explanation.",
                icon=":material/info:",
            )

        with st.expander("Decision path", icon=":material/route:"):
            decision_path = result.get("decision_path")
            if decision_path:
                st.json(decision_path)
            else:
                st.caption("No decision path available for this model.")

        with st.expander("Feature importance", icon=":material/bar_chart:"):
            feature_importance = result.get("feature_importance")
            if feature_importance is not None:
                st.dataframe(
                    feature_importance, width="stretch", hide_index=True
                )
            else:
                st.caption("No feature importance available for this model.")

        local_explanation = result.get("local_explanation")
        if local_explanation:
            with st.expander(
                "Local explanation (SHAP)", icon=":material/bar_chart_4_bars:"
            ):
                st.write(
                    f"Base Value: **{local_explanation['base_value']:.4f}** | "
                    f"Predicted Value: **{local_explanation['prediction_value']:.4f}**"
                )
                import pandas as pd
                df_local = pd.DataFrame(local_explanation["contributions"])
                st.dataframe(df_local, width="stretch", hide_index=True)

        counterfactual = result.get("counterfactual")
        if counterfactual:
            with st.expander(
                "Counterfactual (what-if)", icon=":material/swap_horiz:"
            ):
                recommendation = counterfactual.get("recommendation")
                if counterfactual.get("already_target"):
                    st.caption("Already predicts the target class.")
                elif recommendation:
                    st.write(
                        f"Change **{recommendation['feature']}** from "
                        f"{recommendation['original_value']} to "
                        f"{recommendation['new_value']} to reach target "
                        f"{counterfactual['target']}."
                    )
                else:
                    st.caption(
                        "No single-feature change reaches the target class."
                    )
                st.json(counterfactual)
