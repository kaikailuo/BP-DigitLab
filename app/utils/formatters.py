"""
格式化工具。

提供各种格式化和显示辅助函数。
"""
from typing import Any

import numpy as np


def to_display_image(preprocessed_image: Any) -> np.ndarray:
    """
    将预处理后的图像转换为可显示的格式。
    
    参数：
        preprocessed_image: 来自模型的预处理图像（可能是张量或 numpy 数组）
    
    返回：
        可显示的 numpy 数组
    """
    if hasattr(preprocessed_image, "detach"):
        array = preprocessed_image.detach().cpu().numpy()
    else:
        array = np.asarray(preprocessed_image)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array.squeeze(0)
    return array


def format_device_choice(value: Any) -> str:
    """
    规范化设备选择显示。
    
    参数：
        value: 设备值 ("cpu" 或 "cuda:0" 等)
    
    返回：
        规范化的设备字符串
    """
    return "cpu" if str(value).strip() == "cpu" else "cuda:0"
