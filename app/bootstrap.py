"""Application bootstrap helpers."""
import streamlit as st


def setup_page_config(page_title: str = "BP DigitLab", layout: str = "wide") -> None:
    """Configure the Streamlit page."""
    st.set_page_config(page_title=page_title, layout=layout)


def _inject_base_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        .predict-digit {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.25rem 0 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_train_page() -> None:
    """Initialize the training page."""
    setup_page_config(page_title="BP DigitLab - 训练模型", layout="wide")
    _inject_base_styles()


def initialize_recognition_page() -> None:
    """Initialize the recognition page."""
    setup_page_config(page_title="BP DigitLab - 交互识别", layout="wide")
    _inject_base_styles()
