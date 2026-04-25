"""
画板输入组件。

负责渲染手写画板。
"""
from typing import Optional

import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


def render_canvas_board() -> Optional[dict]:
    """
    渲染手写画板。
    
    返回：
        画板数据字典，如果未安装 streamlit-drawable-canvas 则返回 None
    """
    if not CANVAS_AVAILABLE:
        st.error("未安装 streamlit-drawable-canvas，请先安装: pip install streamlit-drawable-canvas")
        return None
    
    st.markdown("### 鼠标手写数字")
    st.caption("黑底白字，建议写在中央。")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=16,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
    )
    
    return canvas_result
