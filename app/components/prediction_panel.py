"""
预测结果显示组件。

负责渲染模型预测结果和概率分布。
"""
from typing import Any, Dict, List

import numpy as np
import streamlit as st

from app.utils.dataframe import format_probabilities_dataframe
from app.utils.formatters import to_display_image


def render_prediction_panel(result: Dict[str, Any]) -> None:
    """
    渲染预测结果面板。
    
    参数：
        result: 预测结果字典（来自 recognition_service）
    """
    st.markdown(
        f"""
        <div class="predict-digit">{result['prediction']}</div>
        """,
        unsafe_allow_html=True,
    )
    
    st.image(
        to_display_image(result["preprocessed_image"]),
        caption="预处理后 28x28 输入",
        width=200,
        clamp=True,
    )
    
    # 显示概率分布
    df = format_probabilities_dataframe(result["probabilities"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.bar_chart(df.set_index("digit"), width="stretch")
