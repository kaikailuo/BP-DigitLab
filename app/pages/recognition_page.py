"""
识别页面编排。

组合 checkpoint 选择、画板、上传和预测显示。
"""
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from app.components.checkpoint_selector import render_checkpoint_selector
from app.components.canvas_board import render_canvas_board
from app.components.upload_panel import render_upload_panel
from app.components.prediction_panel import render_prediction_panel
from app.components.debug_gallery import render_debug_gallery
from app.services.model_service import get_cached_model_bundle
from app.services.recognition_service import predict_canvas_with_model, predict_uploaded_image_with_model
from app.state import (
    get_loaded_checkpoint,
    get_loaded_device,
    set_loaded_checkpoint,
    set_loaded_device,
    get_ui_save_dir,
    get_ui_result_dir,
    increment_canvas_nonce,
    get_canvas_nonce,
    set_ui_page,
)


def render_recognition_page(workspace_root: Path) -> None:
    """
    渲染识别页面主体。
    
    参数：
        workspace_root: 工作空间根目录
    """
    st.title("BP 手写数字识别实验 - 交互识别页面")
    st.caption("使用鼠标手写数字并识别；无需重新训练")

    if st.button("返回训练模型页面", key="back-to-train", width="stretch"):
        set_ui_page("train")
        st.switch_page("streamlit_app.py")

    save_dir = get_ui_save_dir() or str((workspace_root / "checkpoints").resolve())
    result_dir = get_ui_result_dir() or str((workspace_root / "results").resolve())
    default_device = get_loaded_device() or "cpu"

    # 选择 checkpoint 和设备
    try:
        selected_checkpoint, runtime_device = render_checkpoint_selector(save_dir, result_dir, default_device)
        
        load_clicked = st.button("加载选中模型")
        if load_clicked:
            set_loaded_checkpoint(selected_checkpoint)
            set_loaded_device(runtime_device)

        active_checkpoint = get_loaded_checkpoint() or selected_checkpoint
        active_device = get_loaded_device() or runtime_device

        # 加载模型
        try:
            model, config, _ = get_cached_model_bundle(active_checkpoint, active_device)
            st.success(f"当前模型: {active_checkpoint}")
            st.json(
                {
                    "experiment_name": config.experiment_name,
                    "feature_type": config.feature_type,
                    "hidden_dim": config.hidden_dim,
                    "input_dim": config.input_dim,
                }
            )
        except Exception as exc:
            st.error(f"模型加载失败: {exc}")
            st.stop()

        # 画板识别
        st.markdown("### 鼠标手写数字")
        st.caption("黑底白字，建议写在中央。")

        board_col, result_col = st.columns([1.1, 0.9], gap="small")
        
        with board_col:
            canvas_result = render_canvas_board()
            if canvas_result is None:
                st.stop()

            clear_clicked = st.button("清空画板", width="stretch")
            predict_clicked = st.button("开始识别", type="primary", width="stretch")

        with result_col:
            result_block = st.container()
            with result_block:
                st.markdown("### 预测结果")
                st.info('点击"开始识别"后，这里会显示识别数字与概率分布。')

        # 上传图片识别
        uploaded_image = render_upload_panel()
        if uploaded_image is not None:
            try:
                upload_result = predict_uploaded_image_with_model(
                    image=uploaded_image,
                    model=model,
                    config=config,
                    device=active_device,
                )
                st.markdown("#### 上传图片识别结果")
                render_prediction_panel(upload_result)
                
                if st.checkbox("显示上传图片预处理调试", value=False):
                    render_debug_gallery(upload_result, uploaded_image)
            except Exception as exc:
                st.error(f"上传图片识别失败: {exc}")

        # 处理画板操作
        if clear_clicked:
            increment_canvas_nonce()
            st.rerun()

        if predict_clicked and canvas_result is not None:
            canvas_image = canvas_result.image_data
            if canvas_image is None:
                st.warning("画板为空，请先写一个数字再识别。")
                st.stop()

            max_ink = (
                float(np.max(canvas_image[:, :, 3]))
                if canvas_image.ndim == 3 and canvas_image.shape[2] >= 4
                else float(np.max(canvas_image))
            )
            if max_ink < 10.0:
                st.warning("画板几乎为空，请先写一个数字。")
                st.stop()

            try:
                result = predict_canvas_with_model(
                    canvas_image=canvas_image,
                    model=model,
                    config=config,
                    device=active_device,
                )
                with result_block:
                    st.empty()  # 清空之前的占位符
                    render_prediction_panel(result)
            except Exception as exc:
                with result_block:
                    st.error(f"识别失败: {exc}")

    except Exception as exc:
        st.error(f"识别页面初始化失败: {exc}")
