#!/usr/bin/env python3
"""Small Streamlit upload app for handwritten A-Z letter prediction.

Run:
    streamlit run scripts/handwriting_upload_app.py
"""

from __future__ import annotations

from PIL import Image
import streamlit as st

from image_to_text_vlm.handwriting_hf import (
    DEFAULT_HANDWRITING_MODEL_ID,
    HandwrittenLetterRecognizer,
    format_predictions,
    preprocess_handwritten_letter,
)


@st.cache_resource
def load_recognizer(model_id: str) -> HandwrittenLetterRecognizer:
    return HandwrittenLetterRecognizer(model_id=model_id)


st.set_page_config(page_title="Handwritten Letter Recognizer", page_icon="✍️")
st.title("Handwritten Letter Recognizer")
st.write("Upload one handwritten A-Z letter. The model will predict the letter and confidence.")

model_id = st.text_input("Hugging Face model", DEFAULT_HANDWRITING_MODEL_ID)
crop = st.checkbox("Auto-crop around ink", value=True)
invert = st.checkbox("Invert colors", value=False, help="Use for light writing on a dark background.")
top_k = st.slider("Top predictions", min_value=1, max_value=10, value=5)

uploaded = st.file_uploader("Upload a handwritten letter image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    processed = preprocess_handwritten_letter(original, crop=crop, invert=invert)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("Model input")
        st.image(processed, use_container_width=True)

    recognizer = load_recognizer(model_id)
    predictions = recognizer.predict(processed, top_k=top_k, crop=False)
    best = predictions[0]

    st.success(f"Predicted letter: {best.letter} — confidence {best.confidence:.2%}")
    st.text(format_predictions(predictions))
