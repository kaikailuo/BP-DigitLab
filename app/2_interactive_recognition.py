from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from experiment_manager import scan_experiments
from inference import load_model_from_checkpoint, predict_canvas_with_model, predict_uploaded_image_with_model

try:
    from streamlit_drawable_canvas import st_canvas

    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent


def _normalize_device_choice(value: object) -> str:
    return "cpu" if str(value).strip() == "cpu" else "cuda:0"


@st.cache_resource(show_spinner=False)
def cached_model_bundle(checkpoint_path: str, device: str):
    return load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)


def _load_checkpoint_candidates(save_dir: str, result_dir: str):
    rows = scan_experiments(checkpoint_root=save_dir, result_root=result_dir)
    return [row for row in rows if row.get("checkpoint_path")]


def _checkpoint_label(row):
    best_val = row.get("best_val_acc")
    best_text = "-" if best_val is None else f"{best_val:.4f}"
    return f"{row['experiment_name']} | {row['checkpoint_file']} | best_val_acc={best_text}"


def _ensure_canvas_state():
    if "canvas_nonce" not in st.session_state:
        st.session_state["canvas_nonce"] = 0


def _show_probabilities(probabilities):
    probability_rows = [{"digit": digit, "probability": float(prob)} for digit, prob in enumerate(probabilities)]
    df = pd.DataFrame(probability_rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.bar_chart(df.set_index("digit"), use_container_width=True)


def _to_display_image(preprocessed_image):
    if hasattr(preprocessed_image, "detach"):
        array = preprocessed_image.detach().cpu().numpy()
    else:
        array = np.asarray(preprocessed_image)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    return array


def _render_prediction_result(result):
    st.markdown(
        f"""
        <div class="predict-digit">{result['prediction']}</div>
        """,
        unsafe_allow_html=True,
    )
    st.image(
        _to_display_image(result["preprocessed_image"]),
        caption="预处理后 28x28 输入",
        width=200,
        clamp=True,
    )
    _show_probabilities(result["probabilities"])


def _render_uploaded_debug(result, original_image):
    st.markdown("#### 预处理调试视图")
    debug_cols = st.columns(3)
    with debug_cols[0]:
        st.image(original_image, caption="原图", use_container_width=True)
        if "uploaded_gray" in result:
            gray = np.asarray(result["uploaded_gray"], dtype=np.float32)
            st.image(gray, caption="灰度图", clamp=True, use_container_width=True)
        if "uploaded_gray_corrected" in result:
            gray_corrected = np.asarray(result["uploaded_gray_corrected"], dtype=np.float32)
            st.image(gray_corrected, caption="背景校正后灰度图", clamp=True, use_container_width=True)
    with debug_cols[1]:
        if "uploaded_fg" in result:
            fg = np.asarray(result["uploaded_fg"], dtype=np.float32)
            st.image(fg, caption="前景增强图 fg", clamp=True, use_container_width=True)
        if "uploaded_raw_mask" in result:
            raw_mask = np.asarray(result["uploaded_raw_mask"], dtype=np.float32)
            st.image(raw_mask, caption="raw_mask", clamp=True, use_container_width=True)
        if "uploaded_mask" in result:
            mask = np.asarray(result["uploaded_mask"], dtype=np.float32)
            st.image(mask, caption="最终 mask", clamp=True, use_container_width=True)
    with debug_cols[2]:
        if "uploaded_crop" in result and np.asarray(result["uploaded_crop"]).size > 0:
            crop = np.asarray(result["uploaded_crop"], dtype=np.float32)
            st.image(crop, caption="裁剪后的前景", clamp=True, use_container_width=True)
        st.image(
            _to_display_image(result["preprocessed_image"]),
            caption="最终 28x28 输入",
            width=200,
            clamp=True,
        )


def render_interactive_page():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] {
            font-size: 1.18rem !important;
            font-weight: 700;
            min-height: 2.7rem;
            padding: 0.8rem 0.9rem !important;
            border-radius: 0.6rem;
        }
        div[data-testid="stButton"] button#back-to-train {
            font-size: 1.05rem;
            font-weight: 600;
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
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

    st.title("BP 手写数字识别实验 - 交互识别页面")
    st.caption("使用鼠标手写数字并识别；无需重新训练")

    if st.button("返回训练模型页面", key="back-to-train", use_container_width=True):
        st.session_state["ui_page"] = "train"
        st.rerun()

    save_dir = st.session_state.get("ui_save_dir", str((WORKSPACE_ROOT / "checkpoints").resolve()))
    result_dir = st.session_state.get("ui_result_dir", str((WORKSPACE_ROOT / "results").resolve()))
    default_device = _normalize_device_choice(st.session_state.get("loaded_device", "cpu"))

    checkpoint_candidates = _load_checkpoint_candidates(save_dir=save_dir, result_dir=result_dir)
    if not checkpoint_candidates:
        st.warning("未找到可用 checkpoint，请先在训练模型页面训练或加载模型。")
        st.stop()

    loaded_checkpoint = st.session_state.get("loaded_checkpoint", "")
    selected_index = 0
    if loaded_checkpoint:
        for i, row in enumerate(checkpoint_candidates):
            if row["checkpoint_path"] == loaded_checkpoint:
                selected_index = i
                break

    selected_index = st.selectbox(
        "选择已训练模型 checkpoint",
        options=list(range(len(checkpoint_candidates))),
        index=selected_index,
        format_func=lambda i: _checkpoint_label(checkpoint_candidates[i]),
    )
    selected_row = checkpoint_candidates[selected_index]
    selected_checkpoint = selected_row["checkpoint_path"]

    col_load, col_device = st.columns([1, 1])
    with col_load:
        load_clicked = st.button("加载选中模型")
    with col_device:
        runtime_device = st.selectbox("推理设备", options=["cpu", "cuda"], index=0 if default_device == "cpu" else 1)

    if load_clicked:
        st.session_state["loaded_checkpoint"] = selected_checkpoint
        st.session_state["loaded_device"] = runtime_device

    active_checkpoint = st.session_state.get("loaded_checkpoint", selected_checkpoint)
    active_device = st.session_state.get("loaded_device", runtime_device)

    try:
        model, config, _ = cached_model_bundle(active_checkpoint, active_device)
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

    if not CANVAS_AVAILABLE:
        st.error("未安装 streamlit-drawable-canvas，请先安装: pip install streamlit-drawable-canvas")
        st.stop()

    _ensure_canvas_state()

    st.markdown("### 鼠标手写数字")
    st.caption("黑底白字，建议写在中央。")

    board_col, result_col = st.columns([1.1, 0.9], gap="small")
    with board_col:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"digit_canvas_{st.session_state['canvas_nonce']}",
        )

        clear_clicked = st.button("清空画板", use_container_width=True)
        predict_clicked = st.button("开始识别", type="primary", use_container_width=True)

    with result_col:
        result_block = st.container()
        with result_block:
            st.markdown("### 预测结果")
            st.info("点击“开始识别”后，这里会显示识别数字与概率分布。")

    with st.expander("查看上传图片识别辅助功能", expanded=False):
        uploaded = st.file_uploader("上传一张手写数字图片", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            file_bytes = uploaded.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            try:
                upload_result = predict_uploaded_image_with_model(
                    image=image,
                    model=model,
                    config=config,
                    device=active_device,
                )
                st.markdown("#### 上传图片识别结果")
                _render_prediction_result(upload_result)
                if st.checkbox("显示上传图片预处理调试", value=False):
                    _render_uploaded_debug(upload_result, image)
            except Exception as exc:
                st.error(f"上传图片识别失败: {exc}")

    if clear_clicked:
        st.session_state["canvas_nonce"] += 1
        st.rerun()

    if predict_clicked:
        canvas_image = canvas_result.image_data
        if canvas_image is None:
            st.warning("画板为空，请先写一个数字再识别。")
            st.stop()

        max_ink = float(np.max(canvas_image[:, :, 3])) if canvas_image.ndim == 3 and canvas_image.shape[2] >= 4 else float(np.max(canvas_image))
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
                _render_prediction_result(result)
        except Exception as exc:
            with result_block:
                st.error(f"识别失败: {exc}")


if __name__ == "__main__":
    st.set_page_config(page_title="BP DigitLab - 交互识别", layout="wide")
    render_interactive_page()
