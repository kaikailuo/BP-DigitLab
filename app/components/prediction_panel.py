"""Prediction result display component."""
from typing import Any, Dict

import streamlit as st

from app.utils.dataframe import format_probabilities_dataframe
from app.utils.formatters import to_display_image


def render_prediction_panel(result: Dict[str, Any]) -> None:
    """Render prediction result and class probability distribution."""
    prediction_label = result.get("prediction_label", str(result["prediction"]))
    st.markdown(
        f"""
        <div class="predict-digit">{prediction_label}</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"类别 ID: {result['prediction']}")

    st.image(
        to_display_image(result["preprocessed_image"]),
        caption="预处理后 28x28 输入",
        width=200,
        clamp=True,
    )

    df = format_probabilities_dataframe(
        result["probabilities"],
        class_names=result.get("class_names"),
    )
    st.dataframe(df, hide_index=True, width="stretch")
    st.bar_chart(df.set_index("class"), width="stretch")
