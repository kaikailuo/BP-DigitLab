import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import torch

from src.datasets import build_shared_stats, normalize_image_tensor
from src.features import get_feature_extractor, standardize_feature
from src.hparams import BPTrainingHparams
from src.models import MLP
from src.utils import resolve_device
from src.inference.image_preprocess import preprocess_canvas_image, preprocess_uploaded_image

_EPS = 1e-6


def _normalize_hparams_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(data or {})
    if "learning_rate" in payload and "lr" not in payload:
        payload["lr"] = payload["learning_rate"]

    allowed = {item.name for item in fields(BPTrainingHparams)}
    return {key: value for key, value in payload.items() if key in allowed}


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
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
    elif input_dim == 28 * 28 + 28 + 28:
        inferred["feature_type"] = "pixel_projection"
    elif input_dim == 28 * 28 + 28 + 28 + 2:
        inferred["feature_type"] = "pixel_projection_profile"

    return inferred


def _recover_config_from_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
) -> BPTrainingHparams:
    data: Dict[str, Any] = {}

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        data.update(checkpoint_config)

    state_dict = checkpoint.get("model_state_dict", {})
    if isinstance(state_dict, dict):
        inferred = _infer_config_from_state_dict(state_dict)
        for key, value in inferred.items():
            data.setdefault(key, value)

    data.setdefault("experiment_name", checkpoint_path.stem.replace("_best", ""))
    normalized = _normalize_hparams_payload(data)
    return BPTrainingHparams(**normalized)


def _resolve_project_root(checkpoint_path: Path) -> Path:
    checkpoint_dir = checkpoint_path.parent
    if checkpoint_dir.name == "checkpoints":
        return checkpoint_dir.parent
    return checkpoint_dir.parent if checkpoint_dir.parent.exists() else checkpoint_dir


def _resolve_result_dir(config: BPTrainingHparams, checkpoint_path: Path) -> Path:
    result_dir = Path(config.result_dir)
    if result_dir.is_absolute():
        return result_dir / config.experiment_name

    project_root = _resolve_project_root(checkpoint_path)
    candidate = project_root / result_dir / config.experiment_name
    if candidate.exists():
        return candidate

    return (Path.cwd() / result_dir / config.experiment_name).resolve()


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _stat_to_tensor(value: Any, device: torch.device, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _mean_std_from_tensor(tensor: torch.Tensor) -> tuple[float, float]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return 0.0, 0.0
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0
    return mean, std


class DigitPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = resolve_device(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"无效 checkpoint：{self.checkpoint_path}")

        self.checkpoint = checkpoint
        self.config = _recover_config_from_checkpoint(checkpoint, self.checkpoint_path)
        self.result_dir = _resolve_result_dir(self.config, self.checkpoint_path)

        self.model = MLP(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            num_classes=self.config.num_classes,
            activation=self.config.activation,
            dropout=self.config.dropout,
            batch_norm=self.config.batch_norm,
            weight_init=self.config.weight_init,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.feature_extractor = get_feature_extractor(self.config.feature_type)
        self.normalization_stats = self._load_normalization_stats()

    def _load_normalization_stats(self) -> Dict[str, Any]:
        checkpoint_stats = self.checkpoint.get("normalization_stats", {})
        data_stats = _load_json_file(self.result_dir / "data_stats.json")

        stats = {
            "source": {
                "checkpoint": bool(checkpoint_stats),
                "data_stats": bool(data_stats),
                "rebuilt": False,
            },
            "normalize_images": bool(self.config.normalize_images),
            "normalize_features": bool(self.config.normalize_features),
            "image_mean": None,
            "image_std": None,
            "feature_mean": None,
            "feature_std": None,
        }

        for key in ("image_mean", "image_std", "feature_mean", "feature_std"):
            value = checkpoint_stats.get(key)
            if value is None:
                value = data_stats.get(key)
            stats[key] = _stat_to_tensor(value, self.device, key)

        need_image_stats = self.config.normalize_images and (
            stats["image_mean"] is None or stats["image_std"] is None
        )
        need_feature_stats = self.config.normalize_features and (
            stats["feature_mean"] is None or stats["feature_std"] is None
        )

        if need_image_stats or need_feature_stats:
            try:
                rebuilt = build_shared_stats(self.config)
            except Exception as exc:
                missing_parts = []
                if need_image_stats:
                    missing_parts.append("image_mean/image_std")
                if need_feature_stats:
                    missing_parts.append("feature_mean/feature_std")
                joined = ", ".join(missing_parts)
                raise RuntimeError(
                    f"无法恢复归一化统计量 {joined}。"
                    f"checkpoint 与 data_stats.json 均缺失，且按训练配置重建失败：{exc}"
                ) from exc

            stats["source"]["rebuilt"] = True
            for key in ("image_mean", "image_std", "feature_mean", "feature_std"):
                if stats[key] is None:
                    stats[key] = _stat_to_tensor(rebuilt.get(key), self.device, key)

        if self.config.normalize_images and (stats["image_mean"] is None or stats["image_std"] is None):
            raise RuntimeError(
                "当前模型配置启用了 image normalization，但未能从 checkpoint、"
                "data_stats.json 或训练配置重建中恢复 image_mean/image_std。"
            )

        if self.config.normalize_features and (
            stats["feature_mean"] is None or stats["feature_std"] is None
        ):
            raise RuntimeError(
                "当前模型配置启用了 feature normalization，但未能从 checkpoint、"
                "data_stats.json 或训练配置重建中恢复 feature_mean/feature_std。"
            )

        return stats

    def _predict_from_raw_tensor(self, raw_image_tensor: torch.Tensor, extra: Dict[str, Any] | None = None):
        raw_image_tensor = raw_image_tensor.detach().clone().to(torch.float32)
        if raw_image_tensor.dim() == 2:
            raw_image_tensor = raw_image_tensor.unsqueeze(0)
        if raw_image_tensor.dim() != 3 or raw_image_tensor.shape[0] != 1:
            raise ValueError("image_tensor 形状应为 [1, 28, 28] 或 [28, 28]。")

        display_tensor = raw_image_tensor.detach().cpu().clone()
        raw_image = raw_image_tensor.to(self.device)
        normalized_image = normalize_image_tensor(
            raw_image,
            self.normalization_stats["image_mean"],
            self.normalization_stats["image_std"],
        )

        feature = self.feature_extractor(normalized_image)
        normalized_feature = standardize_feature(
            feature,
            self.normalization_stats["feature_mean"],
            self.normalization_stats["feature_std"],
        )

        with torch.no_grad():
            logits = self.model(normalized_feature.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
            prediction = int(logits.argmax(dim=1).item())

        raw_image_mean, raw_image_std = _mean_std_from_tensor(raw_image_tensor)
        norm_image_mean, norm_image_std = _mean_std_from_tensor(normalized_image)
        feature_mean, feature_std = _mean_std_from_tensor(normalized_feature)

        result = {
            "prediction": prediction,
            "probabilities": probabilities,
            "config": self.config,
            "preprocessed_image": display_tensor,
            "debug": {
                "feature_type": self.config.feature_type,
                "input_dim": self.config.input_dim,
                "normalize_images": bool(self.config.normalize_images),
                "normalize_features": bool(self.config.normalize_features),
                "image_mean_exists": self.normalization_stats["image_mean"] is not None,
                "image_std_exists": self.normalization_stats["image_std"] is not None,
                "feature_mean_exists": self.normalization_stats["feature_mean"] is not None,
                "feature_std_exists": self.normalization_stats["feature_std"] is not None,
                "raw_image_mean": raw_image_mean,
                "raw_image_std": raw_image_std,
                "normalized_image_mean": norm_image_mean,
                "normalized_image_std": norm_image_std,
                "feature_shape": list(normalized_feature.shape),
                "feature_mean_after_norm": feature_mean,
                "feature_std_after_norm": feature_std,
                "normalization_source": self.normalization_stats["source"],
                "normalized_image": normalized_image.detach().cpu().clone(),
            },
        }

        if extra:
            result.update(extra)
        return result

    def predict_tensor(self, image_tensor: torch.Tensor):
        return self._predict_from_raw_tensor(image_tensor)

    def predict_canvas(self, canvas_image):
        preprocess = preprocess_canvas_image(canvas_image)
        extra = {
            "canvas_gray": preprocess["canvas_gray"],
        }
        return self._predict_from_raw_tensor(preprocess["tensor"], extra=extra)

    def predict_uploaded_image(self, image):
        preprocess = preprocess_uploaded_image(image)
        extra = {
            "uploaded_gray": preprocess["gray"],
            "uploaded_gray_corrected": preprocess["gray_corrected"],
            "uploaded_fg": preprocess["fg"],
            "uploaded_raw_mask": preprocess["raw_mask"],
            "uploaded_mask": preprocess["mask"],
            "uploaded_crop": preprocess["crop"],
            "uploaded_final_28": preprocess["final_28"],
        }
        return self._predict_from_raw_tensor(preprocess["tensor"], extra=extra)
