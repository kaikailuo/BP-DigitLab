"""
模型服务。

负责 checkpoint 加载、配置恢复、模型缓存等。
"""
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st
import torch

from src.hparams import BPTrainingHparams
from src.models import MLP
from src.utils import resolve_device


def _normalize_hparams_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """规范化超参数字典，过滤无效字段。"""
    payload = dict(data or {})

    if "learning_rate" in payload and "lr" not in payload:
        payload["lr"] = payload["learning_rate"]

    allowed = {item.name for item in fields(BPTrainingHparams)}
    return {key: value for key, value in payload.items() if key in allowed}


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    从模型权重字典推断配置（input_dim、hidden_dims、num_classes）。
    
    用于 checkpoint 中缺少 config 元数据时的备选方案。
    """
    linear_weights: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith("network.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if not isinstance(value, torch.Tensor) or value.dim() != 2:
            continue
        linear_weights.append((layer_idx, value))

    if not linear_weights:
        return {}

    linear_weights.sort(key=lambda item: item[0])
    first_weight = linear_weights[0][1]
    last_weight = linear_weights[-1][1]

    hidden_dims = [int(weight.shape[0]) for _, weight in linear_weights[:-1]]
    hidden_dim = hidden_dims[0] if hidden_dims else int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    num_classes = int(last_weight.shape[0])

    inferred = {
        "hidden_dim": hidden_dim,
        "hidden_dims": hidden_dims if hidden_dims else [hidden_dim],
        "num_classes": num_classes,
    }

    if input_dim == 28 * 28:
        inferred["feature_type"] = "pixel"
        inferred["projection_dim"] = 28
    elif input_dim > 28 * 28 and (input_dim - 28 * 28) % 2 == 0:
        inferred["feature_type"] = "pixel_projection"
        inferred["projection_dim"] = (input_dim - 28 * 28) // 2

    return inferred


def recover_config_from_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: str,
    fallback: Optional[BPTrainingHparams] = None,
) -> BPTrainingHparams:
    """
    从 checkpoint 恢复配置。
    
    优先使用 checkpoint 中的 config，其次推断，最后使用 fallback。
    
    参数：
        checkpoint: 加载的 checkpoint 字典
        checkpoint_path: checkpoint 文件路径
        fallback: 备选配置
    
    返回：
        BPTrainingHparams 实例
    """
    data: Dict[str, Any] = {}
    if fallback is not None:
        data.update(fallback.to_dict())

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        data.update(checkpoint_config)

    state_dict = checkpoint.get("model_state_dict", {})
    if isinstance(state_dict, dict):
        inferred = _infer_config_from_state_dict(state_dict)
        for key, value in inferred.items():
            data.setdefault(key, value)

    data.setdefault("experiment_name", Path(checkpoint_path).stem.replace("_best", ""))
    data.setdefault("save_dir", str(Path(checkpoint_path).resolve().parent))
    data["checkpoint_path_override"] = str(Path(checkpoint_path).resolve())

    normalized = _normalize_hparams_payload(data)
    return BPTrainingHparams(**normalized)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    fallback_config: Optional[BPTrainingHparams] = None,
) -> Tuple[MLP, BPTrainingHparams, Dict[str, Any]]:
    """
    从 checkpoint 加载模型。
    
    参数：
        checkpoint_path: checkpoint 文件路径
        device: 推理设备 ("cpu" 或 "cuda:0")
        fallback_config: 备选配置
    
    返回：
        (model, config, checkpoint) 元组
    
    异常：
        FileNotFoundError: 文件不存在
        ValueError: checkpoint 格式错误
    """
    abs_path = str(Path(checkpoint_path).resolve())
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Checkpoint not found: {abs_path}")

    torch_device = resolve_device(device)
    checkpoint = torch.load(abs_path, map_location=torch_device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format. Missing 'model_state_dict'.")

    config = recover_config_from_checkpoint(checkpoint, abs_path, fallback=fallback_config)
    model = MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        activation=config.activation,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        weight_init=config.weight_init,
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint


@st.cache_resource(show_spinner=False)
def get_cached_model_bundle(checkpoint_path: str, device: str) -> Tuple[MLP, BPTrainingHparams, Dict[str, Any]]:
    """
    从缓存获取模型包。
    
    Streamlit @st.cache_resource 装饰器确保模型在会话期间只加载一次。
    
    参数：
        checkpoint_path: checkpoint 文件路径
        device: 推理设备
    
    返回：
        (model, config, checkpoint) 元组
    """
    return load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
