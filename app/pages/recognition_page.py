"""Recognition page composition."""
from pathlib import Path

import numpy as np
import streamlit as st

from app.components.canvas_board import render_canvas_board
from app.components.prediction_panel import render_prediction_panel
from app.components.upload_panel import render_upload_panel
from app.services.predictor_service import get_cached_predictor
from app.state import (
    get_loaded_checkpoint,
    get_loaded_device,
    get_recognition_result,
    increment_canvas_nonce,
    set_recognition_result,
    set_ui_page,
)


def _render_model_info(experiment_name: str) -> None:
    st.subheader("当前模型")
    st.info(f"实验名: {experiment_name}")


def _render_result_panel() -> None:
    st.subheader("预测结果")
    result = get_recognition_result()
    if result is None:
        st.info("请先在画板写字符，或上传一张图片，再进行识别。")
        return
    render_prediction_panel(result)


def _is_canvas_empty(canvas_image: np.ndarray) -> bool:
    if canvas_image.ndim == 3 and canvas_image.shape[2] >= 4:
        max_ink = float(np.max(canvas_image[:, :, 3]))
    else:
        max_ink = float(np.max(canvas_image))
    return max_ink < 10.0


def render_recognition_page(workspace_root: Path) -> None:
    """Render the recognition page."""
    del workspace_root

    st.title("BP 手写字符识别实验 - 交互识别页面")
    st.caption("识别页只使用训练页已加载的模型。")

    if st.button("返回训练界面", key="back-to-train", width="stretch"):
        set_ui_page("train")
        st.switch_page("streamlit_app.py")

    checkpoint_path = get_loaded_checkpoint()
    device = get_loaded_device()
    if not checkpoint_path:
        st.warning("当前没有已加载模型，请返回训练页选择历史模型后再进入识别页。")
        st.stop()
    if not device:
        st.warning("当前没有已加载设备信息，请返回训练页重新进入识别页。")
        st.stop()

    try:
        predictor = get_cached_predictor(checkpoint_path, device)
    except Exception as exc:
        st.error(f"模型加载失败: {exc}")
        st.stop()

    _render_model_info(predictor.config.experiment_name)

    input_mode = st.radio(
        "输入方式",
        options=("画板输入", "图片上传"),
        horizontal=True,
        index=0,
    )

    input_col, result_col = st.columns([1.1, 0.9], gap="large")

    with input_col:
        st.subheader("输入区域")
        if input_mode == "画板输入":
            canvas_result = render_canvas_board()
            clear_clicked = st.button("清空画板", width="stretch")
            predict_clicked = st.button("开始识别", type="primary", width="stretch")

            if clear_clicked:
                increment_canvas_nonce()
                set_recognition_result(None)
                st.rerun()

            if predict_clicked:
                if canvas_result is None or canvas_result.image_data is None:
                    st.warning("画板为空，请先写一个字符再识别。")
                elif _is_canvas_empty(canvas_result.image_data):
                    st.warning("画板几乎为空，请先写一个字符。")
                else:
                    try:
                        result = predictor.predict_canvas(canvas_result.image_data)
                        set_recognition_result(result)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"识别失败: {exc}")
        else:
            uploaded_image = render_upload_panel()
            predict_clicked = st.button("开始识别", type="primary", width="stretch")
            if predict_clicked:
                if uploaded_image is None:
                    st.warning("请先上传一张图片。")
                else:
                    try:
                        result = predictor.predict_uploaded_image(uploaded_image)
                        set_recognition_result(result)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"上传图片识别失败: {exc}")

    with result_col:
        _render_result_panel()
