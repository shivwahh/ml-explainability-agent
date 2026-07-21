"""Routing page — config-driven intent classification for a question."""

import streamlit as st

from _shared import get_context
from tools.routing.intent_classifier import IntentClassifier

ctx = get_context()
routing = ctx.routing

st.subheader("Intent classification")
st.caption(
    "The classifier maps a question to an explanation branch using the "
    "keyword map below."
)

classifier = IntentClassifier(
    intents=routing["intents"],
    default_intent=routing["default_intent"],
)
question = st.text_input(
    "Ask a question",
    value="Why did the model choose this class?",
)
if question:
    with st.container(border=True):
        st.metric("Resolved intent", classifier.classify(question))

# Keyword map as a table instead of a raw JSON dump.
st.subheader("Intent keyword map")
st.dataframe(
    [
        {"intent": intent, "keywords": ", ".join(keywords)}
        for intent, keywords in routing["intents"].items()
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "intent": st.column_config.TextColumn("Intent"),
        "keywords": st.column_config.TextColumn("Keywords", width="large"),
    },
)
st.caption(f"Default intent: `{routing['default_intent']}`")
