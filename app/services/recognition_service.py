"""
识别服务。

负责图像预处理和模型预测的全流程。
这是从原 inference.py 中提取的与识别相关的所有逻辑。
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from src.features import get_feature_extractor
from src.hparams import BPTrainingHparams
from src.models import MLP
from src.utils import resolve_device


# ============== 画板图像处理 ==============
def _extract_canvas_gray(canvas_image: np.ndarray) -> np.ndarray:
    """将 Streamlit 画板转换为灰度浮点图像 [0, 255]。"""
    if canvas_image is None:
        raise ValueError("Canvas image is empty.")

    if not isinstance(canvas_image, np.ndarray):
        raise TypeError("canvas_image must be a numpy array.")

    arr = canvas_image
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        channels = arr.shape[2]
        if channels >= 4:
            rgb = arr[:, :, :3].astype(np.float32)
            alpha = arr[:, :, 3].astype(np.float32) / 255.0
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            gray = gray * alpha
        elif channels == 3:
            rgb = arr.astype(np.float32)
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        else:
            raise ValueError("Unsupported canvas image channel count.")
    else:
        raise ValueError("Unsupported canvas image shape.")

    return np.clip(gray, 0.0, 255.0).astype(np.float32)


# ============== 上传图像处理 ==============
def _extract_uploaded_gray(image: Image.Image) -> np.ndarray:
    """将上传的 PIL 图像转换为灰度浮点数组 [0, 255]。"""
    if image is None:
        raise ValueError("Input image is empty.")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    image = ImageOps.exif_transpose(image)

    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("L")
    else:
        image = image.convert("L")

    image = image.filter(ImageFilter.MedianFilter(size=3))
    return np.asarray(image, dtype=np.float32)


# ============== 前景提取 ==============
def _border_pixels(gray_0_255: np.ndarray) -> np.ndarray:
    """提取图像边界像素。"""
    h, w = gray_0_255.shape
    if h < 3 or w < 3:
        return gray_0_255.reshape(-1)

    return np.concatenate(
        [
            gray_0_255[0, :],
            gray_0_255[-1, :],
            gray_0_255[:, 0],
            gray_0_255[:, -1],
        ],
        axis=0,
    )


def _to_foreground_high(gray_0_255: np.ndarray) -> np.ndarray:
    """确保数字笔画是高强度，背景是低强度。"""
    border = _border_pixels(gray_0_255)
    bg_level = float(np.median(border))

    high_score = float(np.percentile(gray_0_255, 99.0)) - bg_level
    low_score = bg_level - float(np.percentile(gray_0_255, 1.0))

    if low_score > high_score:
        fg = 255.0 - gray_0_255
    else:
        fg = gray_0_255

    return np.clip(fg / 255.0, 0.0, 1.0).astype(np.float32)


def _binarize_foreground(fg_0_1: np.ndarray) -> np.ndarray:
    """二值化前景。"""
    non_zero = fg_0_1[fg_0_1 > 0.0]
    if non_zero.size == 0:
        return np.zeros_like(fg_0_1, dtype=bool)

    p50 = float(np.percentile(non_zero, 50.0))
    max_v = float(non_zero.max())
    threshold = max(0.12, min(0.35, max(0.18 * max_v, p50)))
    return fg_0_1 >= threshold


def _remove_background_shading(gray_0_255: np.ndarray, blur_radius: float = 15.0) -> np.ndarray:
    """从上传的图像中估计并去除大规模背景阴影。"""
    clipped = np.clip(gray_0_255, 0.0, 255.0).astype(np.uint8)
    pil_gray = Image.fromarray(clipped, mode="L")
    blurred = pil_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = np.asarray(blurred, dtype=np.float32)

    base_level = float(np.percentile(bg, 70.0))
    corrected = gray_0_255.astype(np.float32) - bg + base_level

    low = float(np.percentile(corrected, 1.5))
    high = float(np.percentile(corrected, 98.5))
    if high - low > 1e-6:
        corrected = (corrected - low) / (high - low) * 255.0

    return np.clip(corrected, 0.0, 255.0).astype(np.float32)


# ============== 阈值与连通域 ==============
def _otsu_threshold_01(image_0_1: np.ndarray) -> float:
    """计算 Otsu 阈值（图像值在 [0, 1] 范围内）。"""
    values = np.clip(image_0_1.astype(np.float32).reshape(-1), 0.0, 1.0)
    if values.size == 0:
        return 0.5

    hist, _ = np.histogram(values, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5

    prob = hist / total
    omega = np.cumsum(prob)
    bin_centers = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b_sq = np.zeros_like(omega)
    valid = denom > 1e-12
    sigma_b_sq[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]

    return float(bin_centers[int(np.argmax(sigma_b_sq))])


def _connected_components(mask: np.ndarray):
    """找到二值图像中的连通域。"""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            coords = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            components.append(coords)

    return components


def _select_main_component(mask: np.ndarray, weight_map: np.ndarray) -> Tuple[np.ndarray, float]:
    """选择最符合单个居中数字的连通域。"""
    h, w = mask.shape
    total_pixels = float(h * w)
    min_area = max(20, int(total_pixels * 0.0008))
    max_area = int(total_pixels * 0.55)

    best_score = -1.0
    best_coords = None

    components = _connected_components(mask)
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    center_norm = max(np.sqrt(center_y * center_y + center_x * center_x), 1e-6)

    for coords in components:
        area = len(coords)
        if area < min_area or area > max_area:
            continue

        ys = np.array([p[0] for p in coords], dtype=np.int32)
        xs = np.array([p[1] for p in coords], dtype=np.int32)
        values = weight_map[ys, xs]
        mass = float(values.sum())
        if mass <= 1e-6:
            continue

        cy = float((ys * values).sum() / mass)
        cx = float((xs * values).sum() / mass)
        center_distance = float(np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2) / center_norm)

        touches_border = bool(
            (ys.min() <= 1)
            or (xs.min() <= 1)
            or (ys.max() >= h - 2)
            or (xs.max() >= w - 2)
        )

        density = mass / float(area)
        area_ratio = area / total_pixels

        border_penalty = 0.55 if touches_border else 1.0
        center_penalty = 1.0 / (1.0 + 2.2 * center_distance * center_distance)
        area_penalty = 1.0 / (1.0 + 3.2 * max(0.0, area_ratio - 0.18))
        score = mass * (0.65 + 0.35 * density) * border_penalty * center_penalty * area_penalty

        if score > best_score:
            best_score = score
            best_coords = coords

    out = np.zeros_like(mask, dtype=bool)
    if best_coords is None:
        return out, 0.0

    ys = np.array([p[0] for p in best_coords], dtype=np.int32)
    xs = np.array([p[1] for p in best_coords], dtype=np.int32)
    out[ys, xs] = True
    return out, float(best_score)


# ============== 掩码处理 ==============
def _foreground_mask_from_gray(gray_0_255: np.ndarray) -> np.ndarray:
    """从灰度图像提取清理后的前景掩码（用于上传图像）。"""
    _, _, _, final_mask = _foreground_mask_from_gray_debug(gray_0_255)
    return final_mask


def _foreground_mask_from_gray_debug(gray_0_255: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (gray_corrected, fg, raw_mask, final_mask)（调试用）。"""
    gray_corrected = _remove_background_shading(gray_0_255)
    fg = _to_foreground_high(gray_corrected)

    non_zero = fg[fg > 0]
    if non_zero.size == 0:
        empty = np.zeros_like(fg, dtype=bool)
        return gray_corrected, fg, empty, empty

    p88 = float(np.percentile(non_zero, 88.0))
    p92 = float(np.percentile(non_zero, 92.0))
    p95 = float(np.percentile(non_zero, 95.0))
    otsu_t = _otsu_threshold_01(fg)

    thresholds = [
        max(0.10, min(0.75, otsu_t)),
        max(0.12, min(0.85, p88)),
        max(0.14, min(0.90, p92)),
        max(0.16, min(0.92, p95)),
    ]

    best_final = np.zeros_like(fg, dtype=bool)
    best_raw = np.zeros_like(fg, dtype=bool)
    best_score = -1.0

    for threshold in thresholds:
        raw_mask = fg >= float(threshold)
        if int(raw_mask.sum()) < 20:
            continue

        final_mask, score = _select_main_component(raw_mask, fg)
        if score > best_score and int(final_mask.sum()) >= 20:
            best_score = score
            best_raw = raw_mask
            best_final = final_mask

    if int(best_final.sum()) < 20:
        raw_mask = fg >= max(0.18, float(np.percentile(non_zero, 90.0)))
        final_mask, _ = _select_main_component(raw_mask, fg)
        best_raw = raw_mask
        best_final = final_mask

    if int(best_final.sum()) < 20:
        empty = np.zeros_like(fg, dtype=bool)
        return gray_corrected, fg, best_raw, empty

    return gray_corrected, fg, best_raw, best_final


# ============== 裁剪与缩放 ==============
def _extract_foreground_bbox(mask: np.ndarray, pad: int = 2) -> Optional[Tuple[int, int, int, int]]:
    """返回最小前景 bbox (y0, y1, x0, x1)。"""
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad)
    return y0, y1, x0, x1


def _extract_foreground_crop(
    gray_0_255: np.ndarray,
    mask: np.ndarray,
    pad: int = 2,
    fg_0_1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """裁剪前景区域并清零裁剪内的背景像素。"""
    bbox = _extract_foreground_bbox(mask, pad=pad)
    if bbox is None:
        return np.zeros((0, 0), dtype=np.float32)

    y0, y1, x0, x1 = bbox
    fg = fg_0_1 if fg_0_1 is not None else _to_foreground_high(gray_0_255)
    crop = fg[y0:y1, x0:x1].astype(np.float32)
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 0.0
    return crop


def _resize_and_pad_to_28x28(crop_0_1: np.ndarray, target_long_side: int = 20) -> np.ndarray:
    """调整大小并填充到 28x28。"""
    canvas = np.zeros((28, 28), dtype=np.float32)
    if crop_0_1.size == 0:
        return canvas

    h, w = crop_0_1.shape
    if h <= 0 or w <= 0:
        return canvas

    scale = float(target_long_side) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized_pil = Image.fromarray(np.clip(crop_0_1 * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized_pil = resized_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized = np.asarray(resized_pil, dtype=np.float32) / 255.0

    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _resize_and_pad_to_28x28_with_mask(
    crop_0_1: np.ndarray,
    mask: np.ndarray,
    target_long_side: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """调整大小并填充，同时保留掩码。"""
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas_mask = np.zeros((28, 28), dtype=bool)
    if crop_0_1.size == 0 or mask.size == 0:
        return canvas, canvas_mask

    h, w = crop_0_1.shape
    if h <= 0 or w <= 0:
        return canvas, canvas_mask

    scale = float(target_long_side) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized_crop = Image.fromarray(np.clip(crop_0_1 * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    resized_crop = resized_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_crop_arr = np.asarray(resized_crop, dtype=np.float32) / 255.0

    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized_mask = resized_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    resized_mask_arr = np.asarray(resized_mask, dtype=np.uint8) > 0

    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized_crop_arr
    canvas_mask[top : top + new_h, left : left + new_w] = resized_mask_arr
    return canvas, canvas_mask


# ============== 中心化 ==============
def _shift_with_zero_pad(img_0_1: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """按指定距离移动图像（零填充）。"""
    out = np.zeros_like(img_0_1, dtype=np.float32)
    h, w = img_0_1.shape

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y) if shift_y >= 0 else h
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x) if shift_x >= 0 else w
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img_0_1[src_y0:src_y1, src_x0:src_x1]

    return out


def _center_by_mass(img_0_1: np.ndarray) -> np.ndarray:
    """通过质量中心来中心化图像。"""
    mass = float(img_0_1.sum())
    if mass <= 1e-8:
        return img_0_1.astype(np.float32)

    ys, xs = np.indices(img_0_1.shape, dtype=np.float32)
    cy = float((ys * img_0_1).sum() / mass)
    cx = float((xs * img_0_1).sum() / mass)

    target_center = 13.5
    shift_y = int(round(target_center - cy))
    shift_x = int(round(target_center - cx))
    return _shift_with_zero_pad(img_0_1.astype(np.float32), shift_y=shift_y, shift_x=shift_x)


def _center_foreground_with_mask(img_0_1: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """通过质量中心来中心化图像和掩码。"""
    mass = float(img_0_1.sum())
    if mass <= 1e-8:
        return img_0_1.astype(np.float32), mask.astype(bool)

    ys, xs = np.indices(img_0_1.shape, dtype=np.float32)
    cy = float((ys * img_0_1).sum() / mass)
    cx = float((xs * img_0_1).sum() / mass)

    target_center = 13.5
    shift_y = int(round(target_center - cy))
    shift_x = int(round(target_center - cx))

    centered_img = _shift_with_zero_pad(img_0_1.astype(np.float32), shift_y=shift_y, shift_x=shift_x)
    centered_mask = _shift_with_zero_pad(mask.astype(np.float32), shift_y=shift_y, shift_x=shift_x) > 0.5
    return centered_img, centered_mask


# ============== 预处理流程 ==============
def _normalize_28x28(gray_image: Image.Image) -> torch.Tensor:
    """规范化灰度 PIL 图像到 28x28 张量。"""
    gray = np.asarray(gray_image.convert("L"), dtype=np.float32)
    fg = _to_foreground_high(gray)
    mask = _binarize_foreground(fg)
    crop = _extract_foreground_crop(gray, mask, pad=1, fg_0_1=fg)

    if crop.size == 0:
        out = np.zeros((28, 28), dtype=np.float32)
    else:
        out = _resize_and_pad_to_28x28(crop, target_long_side=20)
        out = _center_by_mass(out)

    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(out).unsqueeze(0)


def preprocess_canvas_image(canvas_image: np.ndarray) -> torch.Tensor:
    """预处理画板图像为 28x28 张量。"""
    gray_0_255 = _extract_canvas_gray(canvas_image)
    pil = Image.fromarray(gray_0_255.astype(np.uint8), mode="L")
    return _normalize_28x28(pil)


def _preprocess_uploaded_image_core(image: Image.Image) -> Dict[str, Any]:
    """预处理上传的 PIL 图像，返回包含调试中间结果的字典。"""
    gray = _extract_uploaded_gray(image)
    gray_corrected, fg, raw_mask, mask = _foreground_mask_from_gray_debug(gray)

    bbox = _extract_foreground_bbox(mask, pad=2)
    if bbox is None:
        final_28 = np.zeros((28, 28), dtype=np.float32)
        return {
            "gray": gray,
            "gray_corrected": gray_corrected,
            "fg": fg,
            "raw_mask": raw_mask,
            "mask": mask,
            "crop": np.zeros((0, 0), dtype=np.float32),
            "final_28": final_28,
            "tensor": torch.from_numpy(final_28).unsqueeze(0),
        }

    crop = _extract_foreground_crop(gray_corrected, mask, pad=2, fg_0_1=fg)
    crop_mask = mask[bbox[0] : bbox[1], bbox[2] : bbox[3]].astype(bool)

    if crop.size == 0 or int(crop_mask.sum()) < 5:
        final_28 = np.zeros((28, 28), dtype=np.float32)
        return {
            "gray": gray,
            "gray_corrected": gray_corrected,
            "fg": fg,
            "raw_mask": raw_mask,
            "mask": mask,
            "crop": np.zeros((0, 0), dtype=np.float32),
            "final_28": final_28,
            "tensor": torch.from_numpy(final_28).unsqueeze(0),
        }

    resized, resized_mask = _resize_and_pad_to_28x28_with_mask(crop, crop_mask, target_long_side=20)
    centered, centered_mask = _center_foreground_with_mask(resized, resized_mask)

    final_28 = np.clip(centered, 0.0, 1.0).astype(np.float32)
    final_28[~centered_mask] = 0.0
    tensor = torch.from_numpy(final_28).unsqueeze(0)

    return {
        "gray": gray,
        "gray_corrected": gray_corrected,
        "fg": fg,
        "raw_mask": raw_mask,
        "mask": mask,
        "crop": crop,
        "final_28": final_28,
        "tensor": tensor,
    }


def preprocess_uploaded_image(image: Image.Image) -> torch.Tensor:
    """预处理上传的 PIL 图像为 28x28 张量。"""
    return _preprocess_uploaded_image_core(image)["tensor"]


# ============== 预测 ==============
def _predict_from_tensor(
    image_tensor: torch.Tensor,
    model: MLP,
    config: BPTrainingHparams,
    device: str = "cpu",
) -> Dict[str, Any]:
    """从张量进行预测。"""
    feature_extractor = get_feature_extractor(feature_type=config.feature_type)

    features = feature_extractor(image_tensor).unsqueeze(0)
    features = features.to(resolve_device(device))

    model.eval()
    with torch.no_grad():
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        prediction = int(logits.argmax(dim=1).item())

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "config": config,
        "preprocessed_image": image_tensor,
    }


def predict_canvas_with_model(
    canvas_image: np.ndarray,
    model: MLP,
    config: BPTrainingHparams,
    device: str = "cpu",
) -> Dict[str, Any]:
    """识别画板图像。"""
    return _predict_from_tensor(
        image_tensor=preprocess_canvas_image(canvas_image),
        model=model,
        config=config,
        device=device,
    )


def predict_uploaded_image_with_model(
    image: Image.Image,
    model: MLP,
    config: BPTrainingHparams,
    device: str = "cpu",
) -> Dict[str, Any]:
    """识别上传的图像，返回包含调试中间结果的预测。"""
    preprocess = _preprocess_uploaded_image_core(image)
    result = _predict_from_tensor(
        image_tensor=preprocess["tensor"],
        model=model,
        config=config,
        device=device,
    )
    result.update(
        {
            "uploaded_gray": preprocess["gray"],
            "uploaded_gray_corrected": preprocess["gray_corrected"],
            "uploaded_fg": preprocess["fg"],
            "uploaded_raw_mask": preprocess["raw_mask"],
            "uploaded_mask": preprocess["mask"],
            "uploaded_crop": preprocess["crop"],
            "uploaded_final_28": preprocess["final_28"],
        }
    )
    return result
