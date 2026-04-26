"""Drawable canvas component for digit recognition."""
from typing import Optional

import streamlit as st

from app.state import get_canvas_nonce

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


def render_canvas_board() -> Optional[dict]:
    """Render the drawing board and return canvas data."""
    if not CANVAS_AVAILABLE:
        st.error(
            "未安装 streamlit-drawable-canvas，请先安装: "
            "pip install streamlit-drawable-canvas"
        )
        return None

    st.markdown("### 鼠标手写数字")
    st.caption("黑底白字，建议尽量写在中间。")

    return st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=16,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"digit_canvas_{get_canvas_nonce()}",
    )
