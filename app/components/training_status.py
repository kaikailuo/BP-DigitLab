"""
训练状态显示组件。

负责渲染训练日志、进度和指标。
"""
from typing import Any, Callable, Dict, List

import streamlit as st


def render_training_status(
    form_data: Dict[str, Any],
    on_start_training: Callable[[Dict[str, Any]], None],
) -> None:
    """
    渲染训练状态区域。
    
    参数：
        form_data: 表单数据
        on_start_training: 训练开始回调函数
    """
    st.subheader("训练状态 / 日志 / 指标")
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    if not form_data.get("start_training", False):
        status_placeholder.info("等待开始训练。")
        return

    # 启动训练流程
    on_start_training(form_data)
