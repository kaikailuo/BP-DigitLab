"""
预设配置服务。

负责读取和管理 YAML 预设文件。
"""
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    加载 YAML 文件。
    
    参数：
        path: YAML 文件路径
    
    返回：
        加载的字典，如果文件不存在或格式错误返回空字典
    """
    if not path.exists() or not path.is_file():
        return {}
    
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def get_preset_list(preset_dir: Path) -> List[str]:
    """
    获取预设文件列表。
    
    参数：
        preset_dir: 预设文件目录
    
    返回：
        预设文件名列表（不含路径）
    """
    if not preset_dir.exists() or not preset_dir.is_dir():
        return []
    
    preset_files = sorted(preset_dir.glob("*.yaml"))
    return [p.name for p in preset_files]


def load_preset(preset_path: Path) -> Dict[str, Any]:
    """
    加载预设配置。
    
    参数：
        preset_path: 预设文件路径
    
    返回：
        预设配置字典
    """
    return load_yaml_file(preset_path)
