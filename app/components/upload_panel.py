"""
图片上传面板组件。

负责渲染图片上传功能。
"""
import io
from typing import Optional

import streamlit as st
from PIL import Image


def render_upload_panel() -> Optional[Image.Image]:
    """
    渲染图片上传面板。
    
    返回：
        上传的 PIL Image 对象，如果未上传则返回 None
    """
    with st.expander("查看上传图片识别辅助功能", expanded=False):
        uploaded = st.file_uploader("上传一张手写数字图片", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            file_bytes = uploaded.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            return image
    
    return None
