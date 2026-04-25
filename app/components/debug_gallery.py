"""
调试图库组件。

负责渲染上传图片的预处理调试视图。
"""
from typing import Any, Dict

import numpy as np
import streamlit as st

from app.utils.formatters import to_display_image


def render_debug_gallery(result: Dict[str, Any], original_image: Any) -> None:
    """
    渲染调试图库。
    
    参数：
        result: 预测结果（包含预处理中间结果）
        original_image: 原始上传图片
    """
    st.markdown("#### 预处理调试视图")
    debug_cols = st.columns(3)
    
    with debug_cols[0]:
        st.image(original_image, caption="原图", width="stretch")
        
        if "uploaded_gray" in result:
            gray = np.asarray(result["uploaded_gray"], dtype=np.float32)
            st.image(gray, caption="灰度图", clamp=True, width="stretch")
        
        if "uploaded_gray_corrected" in result:
            gray_corrected = np.asarray(result["uploaded_gray_corrected"], dtype=np.float32)
            st.image(gray_corrected, caption="背景校正后灰度图", clamp=True, width="stretch")
    
    with debug_cols[1]:
        if "uploaded_fg" in result:
            fg = np.asarray(result["uploaded_fg"], dtype=np.float32)
            st.image(fg, caption="前景增强图 fg", clamp=True, width="stretch")
        
        if "uploaded_raw_mask" in result:
            raw_mask = np.asarray(result["uploaded_raw_mask"], dtype=np.float32)
            st.image(raw_mask, caption="raw_mask", clamp=True, width="stretch")
        
        if "uploaded_mask" in result:
            mask = np.asarray(result["uploaded_mask"], dtype=np.float32)
            st.image(mask, caption="最终 mask", clamp=True, width="stretch")
    
    with debug_cols[2]:
        if "uploaded_crop" in result and np.asarray(result["uploaded_crop"]).size > 0:
            crop = np.asarray(result["uploaded_crop"], dtype=np.float32)
            st.image(crop, caption="裁剪后的前景", clamp=True, width="stretch")
        
        st.image(
            to_display_image(result["preprocessed_image"]),
            caption="最终 28x28 输入",
            width=200,
            clamp=True,
        )
