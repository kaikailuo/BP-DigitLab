"""
Unified Session State Management for Streamlit.
"""
from typing import Any, Dict, Optional
import streamlit as st

# ============== Session State Keys ==============
FORM_KEY_PREFIX = "cfg_"
LOADED_CHECKPOINT = "loaded_checkpoint"
LOADED_DEVICE = "loaded_device"
LOADED_CONFIG = "loaded_config"
UI_PAGE = "ui_page"
UI_SAVE_DIR = "ui_save_dir"
UI_RESULT_DIR = "ui_result_dir"
CANVAS_NONCE = "canvas_nonce"
UPLOAD_PREDICTION_RESULT = "upload_prediction_result"
UPLOAD_ORIGINAL_IMAGE = "upload_original_image"
RECOGNITION_RESULT = "recognition_result"
FORM_INITIALIZED = "cfg_initialized"

# ============== Initialization ==============
def ensure_form_initialized(defaults: Dict[str, Any]) -> None:
    for key, value in defaults.items():
        session_key = f"{FORM_KEY_PREFIX}{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value
    st.session_state[FORM_INITIALIZED] = True

def ensure_recognition_state_initialized() -> None:
    if CANVAS_NONCE not in st.session_state:
        st.session_state[CANVAS_NONCE] = 0
    if UPLOAD_PREDICTION_RESULT not in st.session_state:
        st.session_state[UPLOAD_PREDICTION_RESULT] = None
    if UPLOAD_ORIGINAL_IMAGE not in st.session_state:
        st.session_state[UPLOAD_ORIGINAL_IMAGE] = None
    if RECOGNITION_RESULT not in st.session_state:
        st.session_state[RECOGNITION_RESULT] = None

def ensure_ui_context_initialized(save_dir: str, result_dir: str) -> None:
    if UI_SAVE_DIR not in st.session_state:
        st.session_state[UI_SAVE_DIR] = save_dir
    if UI_RESULT_DIR not in st.session_state:
        st.session_state[UI_RESULT_DIR] = result_dir

# ============== Accessors ==============
def get_form_value(key: str, default: Any = None) -> Any:
    session_key = f"{FORM_KEY_PREFIX}{key}"
    return st.session_state.get(session_key, default)

def set_form_value(key: str, value: Any) -> None:
    session_key = f"{FORM_KEY_PREFIX}{key}"
    st.session_state[session_key] = value

def update_form_values(values: Dict[str, Any]) -> None:
    for key, value in values.items():
        set_form_value(key, value)

def get_loaded_checkpoint() -> Optional[str]:
    return st.session_state.get(LOADED_CHECKPOINT)

def set_loaded_checkpoint(checkpoint_path: str) -> None:
    st.session_state[LOADED_CHECKPOINT] = checkpoint_path

def get_loaded_device() -> Optional[str]:
    return st.session_state.get(LOADED_DEVICE)

def set_loaded_device(device: str) -> None:
    st.session_state[LOADED_DEVICE] = device

def get_loaded_config() -> Optional[Dict[str, Any]]:
    return st.session_state.get(LOADED_CONFIG)

def set_loaded_config(config: Dict[str, Any]) -> None:
    st.session_state[LOADED_CONFIG] = config

def get_ui_page() -> str:
    return st.session_state.get(UI_PAGE, "train")

def set_ui_page(page: str) -> None:
    st.session_state[UI_PAGE] = page

def get_ui_save_dir() -> str:
    return st.session_state.get(UI_SAVE_DIR, "checkpoints")

def set_ui_save_dir(save_dir: str) -> None:
    st.session_state[UI_SAVE_DIR] = save_dir

def get_ui_result_dir() -> str:
    return st.session_state.get(UI_RESULT_DIR, "results")

def set_ui_result_dir(result_dir: str) -> None:
    st.session_state[UI_RESULT_DIR] = result_dir

def get_canvas_nonce() -> int:
    return st.session_state.get(CANVAS_NONCE, 0)

def increment_canvas_nonce() -> None:
    st.session_state[CANVAS_NONCE] = get_canvas_nonce() + 1

def get_upload_prediction_result() -> Optional[Dict[str, Any]]:
    return st.session_state.get(UPLOAD_PREDICTION_RESULT)

def set_upload_prediction_result(result: Dict[str, Any]) -> None:
    st.session_state[UPLOAD_PREDICTION_RESULT] = result

def get_upload_original_image() -> Optional[Any]:
    return st.session_state.get(UPLOAD_ORIGINAL_IMAGE)

def set_upload_original_image(image: Any) -> None:
    st.session_state[UPLOAD_ORIGINAL_IMAGE] = image

def get_recognition_result() -> Optional[Dict[str, Any]]:
    return st.session_state.get(RECOGNITION_RESULT)

def set_recognition_result(result: Optional[Dict[str, Any]]) -> None:
    st.session_state[RECOGNITION_RESULT] = result
