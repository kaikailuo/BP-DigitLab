"""
表单相关服务。

负责表单数据验证、转换、默认值生成，以及超参数转换。
"""
import ast
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.hparams import BPTrainingHparams
from app.schemas import FormData


def parse_hidden_dims(hidden_dims_text: Any) -> List[int]:
    """
    解析 hidden_dims 字符串为整数列表。
    
    支持多种格式：
    - "256,128"
    - "[256, 128]"
    - "256"
    """
    if isinstance(hidden_dims_text, (list, tuple)):
        dims_source = hidden_dims_text
    else:
        text = str(hidden_dims_text).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text
            dims_source = parsed if isinstance(parsed, (list, tuple)) else text
        else:
            dims_source = text

    if isinstance(dims_source, (list, tuple)):
        dims = [str(item).strip() for item in dims_source if str(item).strip()]
    else:
        dims = [item.strip() for item in str(dims_source).split(",") if item.strip()]

    if not dims:
        raise ValueError("hidden_dims 不能为空。")
    
    parsed = [int(item) for item in dims]
    if any(dim <= 0 for dim in parsed):
        raise ValueError("所有 hidden_dims 必须为正整数。")
    
    return parsed


def _normalize_hparams_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化超参数字典，过滤无效字段。
    
    参数：
        data: 原始数据字典
    
    返回：
        仅包含有效字段的字典
    """
    payload = dict(data or {})

    # 兼容别名
    if "learning_rate" in payload and "lr" not in payload:
        payload["lr"] = payload["learning_rate"]

    allowed_fields = {item.name for item in fields(BPTrainingHparams)}
    return {key: value for key, value in payload.items() if key in allowed_fields}


def build_hparams_from_form(form_data: FormData) -> BPTrainingHparams:
    """
    从表单数据构建超参数配置。
    
    参数：
        form_data: 表单数据字典
    
    返回：
        BPTrainingHparams 实例
    """
    hidden_dims = parse_hidden_dims(str(form_data.get("hidden_dims", "256")))

    data: Dict[str, Any] = {
        "experiment_name": str(form_data.get("experiment_name", "")).strip(),
        "epochs": int(form_data.get("epochs", 10)),
        "batch_size": int(form_data.get("batch_size", 64)),
        "lr": float(form_data.get("learning_rate", 0.001)),
        "hidden_dim": hidden_dims[0],
        "hidden_dims": str(form_data.get("hidden_dims", "256")).strip(),
        "dropout": float(form_data.get("dropout", 0.0)),
        "optimizer": str(form_data.get("optimizer", "adam")),
        "seed": int(form_data.get("seed", 42)),
        "feature_type": str(form_data.get("feature_type", "pixel")),
        "train_size": int(form_data.get("train_size", 50000)),
        "val_size": int(form_data.get("val_size", 10000)),
        "test_size": int(form_data.get("test_size", 10000)),
        "momentum": float(form_data.get("momentum", 0.9)),
        "weight_decay": float(form_data.get("weight_decay", 0.0)),
        "scheduler": str(form_data.get("scheduler", "none")),
        "step_size": int(form_data.get("step_size", 10)),
        "gamma": float(form_data.get("gamma", 0.1)),
        "early_stopping": bool(form_data.get("early_stopping", False)),
        "patience": int(form_data.get("patience", 5)),
        "device": str(form_data.get("device", "cpu")),
        "data_root": str(form_data.get("data_root", "./data")),
        "save_dir": str(form_data.get("save_dir", "checkpoints")),
        "result_dir": str(form_data.get("result_dir", "results")),
        "num_workers": int(form_data.get("num_workers", 0)),
        "num_classes": 10,
    }

    # 如果提供了 checkpoint_path，记录覆盖
    checkpoint_path = str(form_data.get("checkpoint_path", "")).strip()
    if checkpoint_path:
        data["checkpoint_path_override"] = str(Path(checkpoint_path).resolve())

    normalized = _normalize_hparams_payload(data)
    return BPTrainingHparams(**normalized)


def get_default_form_values(workspace_root: Path) -> Dict[str, Any]:
    """
    获取默认表单值。
    
    参数：
        workspace_root: 工作空间根目录
    
    返回：
        包含默认值的字典
    """
    base = BPTrainingHparams().to_dict()
    experiment_name = base["experiment_name"]
    save_dir = str((workspace_root / base["save_dir"]).resolve())

    return {
        "experiment_name": experiment_name,
        "epochs": base["epochs"],
        "batch_size": base["batch_size"],
        "learning_rate": base["lr"],
        "hidden_dims": str(base["hidden_dim"]),
        "dropout": base.get("dropout", 0.0),
        "optimizer": base["optimizer"],
        "seed": base["seed"],
        "feature_type": base["feature_type"],
        "checkpoint_path": str(Path(save_dir) / f"{experiment_name}_best.pth"),
        "auto_checkpoint_path": True,
        "train_size": base["train_size"],
        "val_size": base["val_size"],
        "test_size": base["test_size"],
        "momentum": base["momentum"],
        "weight_decay": base["weight_decay"],
        "scheduler": base["scheduler"],
        "step_size": base["step_size"],
        "gamma": base["gamma"],
        "early_stopping": base["early_stopping"],
        "patience": base["patience"],
        "device": _normalize_device_choice(base["device"]),
        "data_root": str((workspace_root / base["data_root"]).resolve()),
        "save_dir": save_dir,
        "result_dir": str((workspace_root / base["result_dir"]).resolve()),
        "num_workers": base["num_workers"],
    }


def _normalize_device_choice(value: Any) -> str:
    """规范化设备选择。"""
    return "cpu" if str(value).strip() == "cpu" else "cuda:0"


def format_hidden_dims_value(value: Any) -> str:
    """
    格式化 hidden_dims 为字符串表示（用于表单显示）。
    """
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = text
        if isinstance(parsed, (list, tuple)):
            return ",".join(str(item).strip() for item in parsed if str(item).strip())
    
    return text
