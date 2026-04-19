import json
import ast
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from src.hparams import BPTrainingHparams


def _normalize_hparams_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data or {})

    if "learning_rate" in payload and "lr" not in payload:
        payload["lr"] = payload["learning_rate"]

    allowed_fields = {item.name for item in fields(BPTrainingHparams)}
    normalized = {key: value for key, value in payload.items() if key in allowed_fields}

    if "hidden_dims" in normalized and normalized["hidden_dims"] is not None:
        normalized["hidden_dims"] = normalized["hidden_dims"]

    return normalized


def parse_hidden_dims(hidden_dims_text: Any) -> List[int]:
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
        raise ValueError("hidden_dims cannot be empty.")
    parsed = [int(item) for item in dims]
    if any(dim <= 0 for dim in parsed):
        raise ValueError("All hidden dims must be positive integers.")
    return parsed


def build_hparams_from_form(form_data: Dict[str, Any]) -> BPTrainingHparams:
    hidden_dims = parse_hidden_dims(str(form_data["hidden_dims"]))

    data: Dict[str, Any] = {
        "experiment_name": str(form_data["experiment_name"]).strip(),
        "epochs": int(form_data["epochs"]),
        "batch_size": int(form_data["batch_size"]),
        "lr": float(form_data["learning_rate"]),
        "hidden_dim": hidden_dims[0],
        "hidden_dims": str(form_data["hidden_dims"]).strip(),
        "dropout": float(form_data["dropout"]),
        "optimizer": str(form_data["optimizer"]),
        "seed": int(form_data["seed"]),
        "feature_type": str(form_data["feature_type"]),
        "train_size": int(form_data["train_size"]),
        "val_size": int(form_data["val_size"]),
        "test_size": int(form_data["test_size"]),
        "momentum": float(form_data["momentum"]),
        "weight_decay": float(form_data["weight_decay"]),
        "scheduler": str(form_data["scheduler"]),
        "step_size": int(form_data["step_size"]),
        "gamma": float(form_data["gamma"]),
        "early_stopping": bool(form_data["early_stopping"]),
        "patience": int(form_data["patience"]),
        "device": str(form_data["device"]),
        "data_root": str(form_data["data_root"]),
        "save_dir": str(form_data["save_dir"]),
        "result_dir": str(form_data["result_dir"]),
        "num_workers": int(form_data["num_workers"]),
        "num_classes": 10,
    }

    checkpoint_path = str(form_data.get("checkpoint_path", "")).strip()
    if checkpoint_path:
        data["checkpoint_path_override"] = str(Path(checkpoint_path).resolve())

    normalized = _normalize_hparams_payload(data)
    return BPTrainingHparams(**normalized)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _safe_read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _safe_checkpoint_meta(path: Path) -> Dict[str, Any]:
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
    name_from_config = str(config.get("experiment_name", "")).strip()
    if name_from_config:
        return name_from_config
    return checkpoint_path.stem.replace("_best", "")


def _build_result_flags(experiment_result_dir: Path) -> Dict[str, Any]:
    history_path = experiment_result_dir / "training_history.json"
    loss_curve_path = experiment_result_dir / "loss_curve.png"
    accuracy_curve_path = experiment_result_dir / "accuracy_curve.png"
    metrics_path = experiment_result_dir / "test_metrics.json"
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
        "has_test_metrics": metrics_path.exists(),
        "has_confusion_matrix": confusion_matrix_path.exists(),
        "has_classification_report": report_path.exists(),
        "history_path": str(history_path.resolve()),
        "loss_curve_path": str(loss_curve_path.resolve()),
        "accuracy_curve_path": str(accuracy_curve_path.resolve()),
        "test_metrics_path": str(metrics_path.resolve()),
        "confusion_matrix_path": str(confusion_matrix_path.resolve()),
        "classification_report_path": str(report_path.resolve()),
    }


def scan_experiments(checkpoint_root: str = "checkpoints", result_root: str = "results") -> List[Dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_root)
    result_dir = Path(result_root)

    rows: List[Dict[str, Any]] = []
    covered_experiments = set()

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
    result_dir = Path(str(experiment_row.get("result_dir", "")))
    if not result_dir.exists() or not result_dir.is_dir():
        return {
            "training_history": {},
            "test_metrics": {},
            "classification_report": "",
            "images": {},
        }

    history_path = result_dir / "training_history.json"
    metrics_path = result_dir / "test_metrics.json"
    report_path = result_dir / "classification_report.txt"

    images = {}
    for name in ["loss_curve.png", "accuracy_curve.png", "confusion_matrix.png", "wrong_samples.png"]:
        path = result_dir / name
        if path.exists() and path.is_file():
            images[name] = str(path.resolve())

    return {
        "training_history": _safe_read_json(history_path),
        "test_metrics": _safe_read_json(metrics_path),
        "classification_report": _safe_read_text(report_path),
        "images": images,
    }


def list_available_checkpoints(checkpoint_root: str = "checkpoints") -> List[str]:
    checkpoint_dir = Path(checkpoint_root)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return []
    return [str(path.resolve()) for path in sorted(checkpoint_dir.glob("*.pth"))]