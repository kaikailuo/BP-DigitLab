from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
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
    if train_size + val_size > total_size:
        raise ValueError(
            f"train_size + val_size = {train_size + val_size} 超过训练集总量 {total_size}。"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(total_size, generator=generator)
    train_indices = shuffled[:train_size]
    val_indices = shuffled[train_size : train_size + val_size]
    return train_indices, val_indices


def _load_subset(base_dataset: MNIST, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for idx in indices.tolist():
        pil_image, label = base_dataset[idx]
        images.append(to_tensor(pil_image))
        labels.append(label)
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
    train_base = MNIST(root=config.data_root, train=True, download=True)
    train_indices, _ = _split_train_val_indices(
        total_size=len(train_base),
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    train_images, _ = _load_subset(train_base, train_indices)

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


class MNISTFeatureDataset(Dataset):
    """在保持原有接口风格的前提下，支持增强和标准化的 MNIST 特征数据集。"""

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

        self.stats_summary = {
            "normalize_images": bool(config.normalize_images),
            "normalize_features": bool(config.normalize_features),
            "image_mean": None if self.image_mean is None else float(self.image_mean.item()),
            "image_std": None if self.image_std is None else float(self.image_std.item()),
        }

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
    train_base = MNIST(root=config.data_root, train=True, download=True)
    train_indices, val_indices = _split_train_val_indices(
        total_size=len(train_base),
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
    )
    shared_stats = _build_shared_stats(config)

    train_images, train_labels = _load_subset(train_base, train_indices)
    val_images, val_labels = _load_subset(train_base, val_indices)

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
    shared_stats = _build_shared_stats(config)
    test_base = MNIST(root=config.data_root, train=False, download=True)
    test_indices = _sample_indices(
        total_size=len(test_base),
        sample_size=config.test_size,
        seed=config.seed + 1,
    )
    test_images, test_labels = _load_subset(test_base, test_indices)

    return MNISTFeatureDataset(
        images=test_images,
        labels=test_labels,
        split="test",
        feature_type=config.feature_type,
        stats=shared_stats,
        config=config,
    )
