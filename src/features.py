from typing import Callable

import torch


FEATURE_DIMS = {
    "pixel": 28 * 28,
    "pixel_projection": 28 * 28 + 28 + 28,
    "pixel_projection_profile": 28 * 28 + 28 + 28 + 2,
}

_EPS = 1e-6


def _ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("输入图像必须为 torch.Tensor。")

    image = image.to(torch.float32)
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.dim() != 3:
        raise ValueError("图像张量形状应为 [1, 28, 28] 或 [28, 28]。")
    return image


def extract_pixel_features(image: torch.Tensor) -> torch.Tensor:
    image = _ensure_image_tensor(image)
    return image.reshape(-1)


def extract_pixel_projection_features(image: torch.Tensor) -> torch.Tensor:
    image = _ensure_image_tensor(image)
    image_2d = image.squeeze(0)

    pixel_feature = image.reshape(-1)
    row_projection = image_2d.sum(dim=1)
    col_projection = image_2d.sum(dim=0)

    return torch.cat([pixel_feature, row_projection, col_projection], dim=0)


def extract_pixel_projection_profile_features(image: torch.Tensor) -> torch.Tensor:
    """在像素和行列投影基础上增加整体墨迹强度与非零像素占比。"""
    image = _ensure_image_tensor(image)
    image_2d = image.squeeze(0)

    base_feature = extract_pixel_projection_features(image)
    ink_mean = image_2d.mean().unsqueeze(0)
    active_ratio = (image_2d > 0.1).to(torch.float32).mean().unsqueeze(0)
    return torch.cat([base_feature, ink_mean, active_ratio], dim=0)


def standardize_feature(
    feature: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if mean is None or std is None:
        return feature
    return (feature - mean) / std.clamp_min(_EPS)


def get_feature_dim(feature_type: str) -> int:
    if feature_type not in FEATURE_DIMS:
        raise ValueError(f"未知 feature_type: {feature_type}")
    return FEATURE_DIMS[feature_type]


def get_feature_extractor(feature_type: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if feature_type == "pixel":
        return extract_pixel_features
    if feature_type == "pixel_projection":
        return extract_pixel_projection_features
    if feature_type == "pixel_projection_profile":
        return extract_pixel_projection_profile_features
    raise ValueError(f"未知 feature_type: {feature_type}")
