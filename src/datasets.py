from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST, MNIST
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms.functional import to_tensor

from src.features import get_feature_extractor, standardize_feature

_EPS = 1e-6


def _sample_indices(total_size: int, sample_size: int, seed: int) -> torch.Tensor:
    if sample_size > total_size:
        raise ValueError(f"采样数量 {sample_size} 不能大于数据总量 {total_size}。")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total_size, generator=generator)[:sample_size]


def _split_train_val_indices(
    total_size: int,
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if train_size is None or val_size is None:
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.2)
    if train_size + val_size > total_size:
        raise ValueError(
            f"train_size + val_size = {train_size + val_size} 超过训练集总量 {total_size}。"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(total_size, generator=generator)
    train_indices = shuffled[:train_size]
    val_indices = shuffled[train_size : train_size + val_size]
    return train_indices, val_indices


def _build_base_dataset(config, train: bool):
    if config.dataset == "mnist":
        return MNIST(root=config.data_root, train=train, download=True)
    if config.dataset == "emnist":
        return EMNIST(
            root=config.data_root,
            split=config.emnist_split,
            train=train,
            download=True,
        )
    raise ValueError(f"不支持的数据集: {config.dataset}")


def _normalize_label(label: int, config) -> int:
    normalized = int(label)
    if config.dataset == "emnist" and config.emnist_split == "letters":
        normalized -= 1
    return normalized


def _load_subset(base_dataset, indices: torch.Tensor, config) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for idx in indices.tolist():
        pil_image, label = base_dataset[idx]
        images.append(to_tensor(pil_image))
        labels.append(_normalize_label(label, config))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _compute_image_stats(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = images.mean()
    std = images.std(unbiased=False).clamp_min(_EPS)
    return mean, std


def _normalize_image(
    image: torch.Tensor,
    image_mean: torch.Tensor | None,
    image_std: torch.Tensor | None,
) -> torch.Tensor:
    if image_mean is None or image_std is None:
        return image
    return (image - image_mean) / image_std.clamp_min(_EPS)


def normalize_image_tensor(
    image: torch.Tensor,
    image_mean: torch.Tensor | None,
    image_std: torch.Tensor | None,
) -> torch.Tensor:
    """对单张图像执行与训练阶段一致的图像标准化。"""
    return _normalize_image(image, image_mean, image_std)


def _compute_feature_stats(
    images: torch.Tensor,
    feature_extractor,
    image_mean: torch.Tensor | None,
    image_std: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    for image in images:
        normalized_image = _normalize_image(image, image_mean, image_std)
        features.append(feature_extractor(normalized_image))
    feature_tensor = torch.stack(features)
    feature_mean = feature_tensor.mean(dim=0)
    feature_std = feature_tensor.std(dim=0, unbiased=False).clamp_min(_EPS)
    return feature_mean, feature_std


def _build_shared_stats(config) -> Dict[str, torch.Tensor | None]:
    train_base = _build_base_dataset(config, train=True)
    train_indices, _ = _split_train_val_indices(
        total_size=len(train_base),
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    train_images, _ = _load_subset(train_base, train_indices, config)

    image_mean, image_std = (None, None)
    if config.normalize_images:
        image_mean, image_std = _compute_image_stats(train_images)

    feature_mean, feature_std = (None, None)
    if config.normalize_features:
        feature_extractor = get_feature_extractor(config.feature_type)
        feature_mean, feature_std = _compute_feature_stats(
            images=train_images,
            feature_extractor=feature_extractor,
            image_mean=image_mean,
            image_std=image_std,
        )

    return {
        "image_mean": image_mean,
        "image_std": image_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def build_shared_stats(config) -> Dict[str, torch.Tensor | None]:
    """暴露共享统计量构建逻辑，供推理阶段复用。"""
    return _build_shared_stats(config)


class MNISTFeatureDataset(Dataset):
    """28x28 手写字符数据集，支持增强与标准化。"""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        split: str,
        feature_type: str,
        stats: Dict[str, torch.Tensor | None],
        config,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split 仅支持 'train'、'val'、'test'。")

        self.split = split
        self.config = config
        self.feature_extractor = get_feature_extractor(feature_type)
        self.raw_images = images.clone()
        self.labels = labels.clone()

        self.image_mean = stats.get("image_mean")
        self.image_std = stats.get("image_std")
        self.feature_mean = stats.get("feature_mean")
        self.feature_std = stats.get("feature_std")

        self.shared_stats = {
            "normalize_images": bool(config.normalize_images),
            "normalize_features": bool(config.normalize_features),
            "image_mean": None if self.image_mean is None else self.image_mean.detach().cpu().clone(),
            "image_std": None if self.image_std is None else self.image_std.detach().cpu().clone(),
            "feature_mean": None
            if self.feature_mean is None
            else self.feature_mean.detach().cpu().clone(),
            "feature_std": None
            if self.feature_std is None
            else self.feature_std.detach().cpu().clone(),
        }
        self.stats_summary = dict(self.shared_stats)

        self.augment = None
        if split == "train" and config.augment_train:
            self.augment = RandomAffine(
                degrees=config.augment_rotation,
                translate=(config.augment_translate, config.augment_translate),
                scale=(config.augment_scale_min, config.augment_scale_max),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.raw_images[index].clone()
        label = self.labels[index]

        if self.augment is not None:
            image = self.augment(image)

        image = _normalize_image(image, self.image_mean, self.image_std)
        feature = self.feature_extractor(image)
        feature = standardize_feature(feature, self.feature_mean, self.feature_std)

        return feature, label


def build_train_val_datasets(config) -> Tuple[MNISTFeatureDataset, MNISTFeatureDataset]:
    train_base = _build_base_dataset(config, train=True)
    if config.train_size is None or config.val_size is None:
        config.train_size = int(len(train_base) * 0.8)
        config.val_size = int(len(train_base) * 0.2)
    train_indices, val_indices = _split_train_val_indices(
        total_size=len(train_base),
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    shared_stats = build_shared_stats(config)

    train_images, train_labels = _load_subset(train_base, train_indices, config)
    val_images, val_labels = _load_subset(train_base, val_indices, config)

    train_set = MNISTFeatureDataset(
        images=train_images,
        labels=train_labels,
        split="train",
        feature_type=config.feature_type,
        stats=shared_stats,
        config=config,
    )
    val_set = MNISTFeatureDataset(
        images=val_images,
        labels=val_labels,
        split="val",
        feature_type=config.feature_type,
        stats=shared_stats,
        config=config,
    )
    return train_set, val_set


def build_test_dataset(config) -> MNISTFeatureDataset:
    shared_stats = build_shared_stats(config)
    test_base = _build_base_dataset(config, train=False)
    if config.test_size is None:
        config.test_size = len(test_base)
    test_indices = _sample_indices(
        total_size=len(test_base),
        sample_size=config.test_size,
        seed=config.seed + 1,
    )
    test_images, test_labels = _load_subset(test_base, test_indices, config)

    return MNISTFeatureDataset(
        images=test_images,
        labels=test_labels,
        split="test",
        feature_type=config.feature_type,
        stats=shared_stats,
        config=config,
    )
