"""
实验历史显示组件。

负责渲染历史实验列表和详情。
"""
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from app.services.experiment_service import scan_experiments, load_experiment_artifacts
from app.utils.dataframe import format_experiment_history_dataframe


def render_experiment_history(
    form_data: Dict[str, Any],
    on_load_model: Optional[callable] = None,
    on_enter_recognition: Optional[callable] = None,
) -> None:
    """
    渲染实验历史区域。
    
    参数：
        form_data: 表单数据
        on_load_model: 加载模型回调
        on_enter_recognition: 进入识别页回调
    """
    st.subheader("历史实验记录")
    
    save_dir = form_data.get("save_dir", "checkpoints")
    result_dir = form_data.get("result_dir", "results")

    if not save_dir or not result_dir:
        st.warning("checkpoints 或 results 目录未配置")
        return

    rows = scan_experiments(
        checkpoint_root=str(save_dir),
        result_root=str(result_dir),
    )

    if not rows:
        st.info("未扫描到历史实验，请先训练至少一次。")
        return

    st.dataframe(format_experiment_history_dataframe(rows), width="stretch", hide_index=True)

    selected_index = st.selectbox(
        "选择历史实验 / 历史模型 checkpoint",
        options=list(range(len(rows))),
        format_func=lambda i: f"{rows[i]['experiment_name']} | {rows[i]['checkpoint_file']}",
    )
    selected_row = rows[selected_index]

    st.caption(f"当前选择: {selected_row['experiment_name']}")

    if selected_row.get("checkpoint_path"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("加载该模型进行识别（无需重新训练）"):
                if on_load_model:
                    on_load_model(selected_row, form_data)
        with col2:
            if st.button("进入交互识别页面"):
                if on_enter_recognition:
                    on_enter_recognition(selected_row, form_data)
    else:
        st.info("该实验只有 results，没有可加载的 checkpoint。")

    artifacts = load_experiment_artifacts(selected_row)

    st.markdown("#### 实验结果可视化")
    image_paths = artifacts.get("images", {})
    if image_paths:
        for image_name, image_path in image_paths.items():
            st.image(image_path, caption=image_name, width="stretch")
    else:
        st.info("未找到可视化图片文件（loss/accuracy/confusion 等）。")

    metrics = artifacts.get("metrics", {})
    if metrics:
        st.markdown("#### metrics.json")
        st.json(metrics)

    report_text = artifacts.get("classification_report", "")
    if report_text:
        st.markdown("#### classification_report.txt")
        st.code(report_text, language="text")
