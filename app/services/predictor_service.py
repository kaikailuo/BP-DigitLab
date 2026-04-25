import streamlit as st

from src.inference import DigitPredictor


@st.cache_resource(show_spinner=False)
def get_cached_predictor(checkpoint_path: str, device: str) -> DigitPredictor:
    return DigitPredictor(checkpoint_path=checkpoint_path, device=device)
