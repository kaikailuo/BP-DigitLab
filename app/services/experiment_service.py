"""
实验管理服务。

负责扫描实验、加载产物、管理 checkpoint 和 results。
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _safe_read_json(path: Path) -> Dict[str, Any]:
    """安全地读取 JSON 文件。"""
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _safe_read_text(path: Path) -> str:
    """安全地读取文本文件。"""
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_checkpoint_meta(path: Path) -> Dict[str, Any]:
    """安全地加载 checkpoint 元数据。"""
    try:
        checkpoint = torch.load(str(path), map_location="cpu")
    except Exception:
        return {}

    if not isinstance(checkpoint, dict):
        return {}

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    return {
        "best_val_acc": checkpoint.get("best_val_acc"),
        "config": config,
        "epoch": checkpoint.get("epoch"),
    }


def _resolve_experiment_name(checkpoint_path: Path, config: Dict[str, Any]) -> str:
    """从 checkpoint 路径或配置中解析实验名称。"""
    name_from_config = str(config.get("experiment_name", "")).strip()
    if name_from_config:
        return name_from_config
    return checkpoint_path.stem.replace("_best", "")


def _build_result_flags(experiment_result_dir: Path) -> Dict[str, Any]:
    """
    构建实验结果标志和文件信息。
    
    返回包含各种产物文件的存在标志和路径的字典。
    """
    history_path = experiment_result_dir / "training_history.json"
    loss_curve_path = experiment_result_dir / "loss_curve.png"
    accuracy_curve_path = experiment_result_dir / "accuracy_curve.png"
    metrics_path = experiment_result_dir / "metrics.json"
    confusion_matrix_path = experiment_result_dir / "confusion_matrix.png"
    report_path = experiment_result_dir / "classification_report.txt"

    history = _safe_read_json(history_path)
    metrics = _safe_read_json(metrics_path)
    val_acc_list = history.get("val_acc", []) if isinstance(history, dict) else []

    return {
        "history": history,
        "metrics": metrics,
        "best_val_acc_from_history": max(val_acc_list) if val_acc_list else None,
        "has_training_history": history_path.exists(),
        "has_loss_curve": loss_curve_path.exists(),
        "has_accuracy_curve": accuracy_curve_path.exists(),
        "has_metrics": metrics_path.exists(),
        "has_confusion_matrix": confusion_matrix_path.exists(),
        "has_classification_report": report_path.exists(),
        "history_path": str(history_path.resolve()),
        "loss_curve_path": str(loss_curve_path.resolve()),
        "accuracy_curve_path": str(accuracy_curve_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "confusion_matrix_path": str(confusion_matrix_path.resolve()),
        "classification_report_path": str(report_path.resolve()),
    }


def scan_experiments(checkpoint_root: str = "checkpoints", result_root: str = "results") -> List[Dict[str, Any]]:
    """
    扫描所有实验（checkpoint 和 results）。
    
    参数：
        checkpoint_root: checkpoint 目录路径
        result_root: results 目录路径
    
    返回：
        实验列表，按更新时间逆序排列
    """
    checkpoint_dir = Path(checkpoint_root)
    result_dir = Path(result_root)

    rows: List[Dict[str, Any]] = []
    covered_experiments = set()

    # 先从 checkpoint 扫描
    checkpoint_files = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []

    for checkpoint_path in checkpoint_files:
        meta = _safe_checkpoint_meta(checkpoint_path)
        config = meta.get("config", {})
        experiment_name = _resolve_experiment_name(checkpoint_path, config)
        covered_experiments.add(experiment_name)

        experiment_result_dir = result_dir / experiment_name
        flags = _build_result_flags(experiment_result_dir)

        best_val_acc = meta.get("best_val_acc")
        if best_val_acc is None:
            best_val_acc = flags["best_val_acc_from_history"]

        row = {
            "experiment_name": experiment_name,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_epoch": meta.get("epoch"),
            "best_val_acc": round(float(best_val_acc), 6) if best_val_acc is not None else None,
            "result_dir": str(experiment_result_dir.resolve()),
            "updated_at": checkpoint_path.stat().st_mtime,
            "source": "checkpoint",
        }
        row.update(flags)
        rows.append(row)

    # 再从 results 扫描未被 checkpoint 覆盖的
    result_dirs = sorted(result_dir.glob("*")) if result_dir.exists() else []
    for experiment_result_dir in result_dirs:
        if not experiment_result_dir.is_dir():
            continue

        experiment_name = experiment_result_dir.name
        if experiment_name in covered_experiments:
            continue

        flags = _build_result_flags(experiment_result_dir)
        best_val_acc = flags["best_val_acc_from_history"]

        row = {
            "experiment_name": experiment_name,
            "checkpoint_file": "-",
            "checkpoint_path": "",
            "checkpoint_epoch": None,
            "best_val_acc": round(float(best_val_acc), 6) if best_val_acc is not None else None,
            "result_dir": str(experiment_result_dir.resolve()),
            "updated_at": experiment_result_dir.stat().st_mtime,
            "source": "result_only",
        }
        row.update(flags)
        rows.append(row)

    rows.sort(key=lambda x: x["updated_at"], reverse=True)
    return rows


def load_experiment_artifacts(experiment_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    加载单个实验的全部产物（图片、报告、指标等）。
    
    参数：
        experiment_row: 从 scan_experiments 获得的单行记录
    
    返回：
        包含产物数据的字典
    """
    result_dir = Path(str(experiment_row.get("result_dir", "")))
    if not result_dir.exists() or not result_dir.is_dir():
        return {
            "training_history": {},
            "metrics": {},
            "classification_report": "",
            "images": {},
        }

    history_path = result_dir / "training_history.json"
    metrics_path = result_dir / "metrics.json"
    report_path = result_dir / "classification_report.txt"

    images = {}
    for name in ["loss_curve.png", "accuracy_curve.png", "confusion_matrix.png", "wrong_samples.png"]:
        path = result_dir / name
        if path.exists() and path.is_file():
            images[name] = str(path.resolve())

    return {
        "training_history": _safe_read_json(history_path),
        "metrics": _safe_read_json(metrics_path),
        "classification_report": _safe_read_text(report_path),
        "images": images,
    }


def list_available_checkpoints(checkpoint_root: str = "checkpoints") -> List[str]:
    """
    列出所有可用的 checkpoint 文件路径。
    
    参数：
        checkpoint_root: checkpoint 目录路径
    
    返回：
        checkpoint 路径列表
    """
    checkpoint_dir = Path(checkpoint_root)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return []
    return [str(path.resolve()) for path in sorted(checkpoint_dir.glob("*.pth"))]
