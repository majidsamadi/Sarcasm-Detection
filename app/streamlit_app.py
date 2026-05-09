#!/usr/bin/env python3
"""Streamlit demo interface for Task 20."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_sarcasm import SarcasmPredictor  # noqa: E402


FINAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_model_config.json"
TASK19_SUMMARY_PATH = PROJECT_ROOT / "reports" / "task19" / "task19_final_model_selection_summary.json"


@st.cache_resource(show_spinner="Loading selected RoBERTa model...")
def load_predictor() -> SarcasmPredictor:
    return SarcasmPredictor(config_path=FINAL_CONFIG_PATH)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


st.set_page_config(
    page_title="Sarcasm Detection Demo",
    page_icon="🧠",
    layout="centered",
)

st.title("Sarcasm Detection Demo")
st.caption("Task 20 demo interface using the selected final model: RoBERTa Version A.")

with st.sidebar:
    st.header("Final Model")
    final_config = load_json(FINAL_CONFIG_PATH)
    task19 = load_json(TASK19_SUMMARY_PATH)

    st.write("**Selected model:** RoBERTa Version A")
    st.write("**Preprocessing:** Version A — stopwords kept")
    st.write("**Input:** text only, using parent comment + comment when context is provided")
    st.write("**Max length:** 128 tokens")

    # These values are from Task 19 final model selection.
    st.metric("Test Accuracy", "0.7223")
    st.metric("Test Macro-F1", "0.7167")

    with st.expander("Repository notes"):
        st.write("Model checkpoints are local-only and are not pushed to GitHub.")
        st.write("This app loads the local checkpoint at `models/roberta/versionA`.")

st.warning(
    "Research demo only. This model should not be used to automatically remove, flag, "
    "or penalize user content because sarcasm depends on context and can be misclassified."
)

st.subheader("Enter Reddit-style text")
parent_comment = st.text_area(
    "Optional parent comment / context",
    height=90,
    placeholder="Example: The weather ruined our picnic plans.",
)
comment = st.text_area(
    "Comment to classify",
    height=120,
    placeholder="Example: Great, exactly what we needed today.",
)

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

st.divider()

with st.expander("Try safe sample inputs"):
    st.markdown(
        "**Likely sarcastic sample:** Parent: `The deadline moved to tomorrow.` Comment: `Perfect, I love surprise deadlines.`"
    )
    st.markdown(
        "**Likely non-sarcastic sample:** Parent: `How was the seminar?` Comment: `It was useful and I learned several new methods.`"
    )

if predict_clicked:
    if not comment.strip() and not parent_comment.strip():
        st.error("Please enter a comment, or parent/comment context.")
    else:
        try:
            predictor = load_predictor()
            result = predictor.predict(comment=comment, parent_comment=parent_comment)

            if result.label == "Sarcastic":
                st.success(f"Prediction: **{result.label}**")
            else:
                st.info(f"Prediction: **{result.label}**")

            st.write(f"**Confidence:** {result.confidence:.4f}")
            st.progress(float(result.confidence))

            p1, p2 = st.columns(2)
            p1.metric("P(non-sarcastic)", f"{result.probability_non_sarcastic:.4f}")
            p2.metric("P(sarcastic)", f"{result.probability_sarcastic:.4f}")

            with st.expander("Combined model input"):
                st.write(result.combined_text)

            with st.expander("Technical details"):
                st.json(result.to_dict())

        except Exception as exc:
            st.error("Prediction failed. Please check that the local model checkpoint exists.")
            st.exception(exc)
