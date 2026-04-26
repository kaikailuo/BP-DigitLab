"""Image upload widget for recognition."""
import io
from typing import Optional

import streamlit as st
from PIL import Image


def render_upload_panel() -> Optional[Image.Image]:
    """Render a plain file uploader and return the uploaded image."""
    uploaded = st.file_uploader(
        "上传一张手写数字图片",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded is None:
        return None

    file_bytes = uploaded.read()
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")
