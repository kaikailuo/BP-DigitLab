"""
Checkpoint 选择器组件。

负责渲染 checkpoint 选择和设备选择。
"""
from typing import Any, Dict, List

import streamlit as st

from app.services.experiment_service import scan_experiments


def render_checkpoint_selector(save_dir: str, result_dir: str, default_device: str = "cpu") -> tuple:
    """
    渲染 checkpoint 选择器。
    
    参数：
        save_dir: checkpoint 保存目录
        result_dir: 结果目录
        default_device: 默认设备
    
    返回：
        (selected_checkpoint_path, selected_device) 元组
    """
    rows = scan_experiments(checkpoint_root=save_dir, result_root=result_dir)
    checkpoint_rows = [row for row in rows if row.get("checkpoint_path")]
    
    if not checkpoint_rows:
        st.warning("未找到可用 checkpoint，请先在训练模型页面训练或加载模型。")
        st.stop()
    
    def checkpoint_label(row):
        best_val = row.get("best_val_acc")
        best_text = "-" if best_val is None else f"{best_val:.4f}"
        return f"{row['experiment_name']} | {row['checkpoint_file']} | best_val_acc={best_text}"
    
    selected_index = st.selectbox(
        "选择已训练模型 checkpoint",
        options=list(range(len(checkpoint_rows))),
        format_func=lambda i: checkpoint_label(checkpoint_rows[i]),
    )
    selected_row = checkpoint_rows[selected_index]
    
    col_device = st.columns(1)[0]
    with col_device:
        runtime_device = st.selectbox(
            "推理设备",
            options=["cpu", "cuda:0"],
            index=0 if default_device == "cpu" else 1
        )
    
    return selected_row["checkpoint_path"], runtime_device
