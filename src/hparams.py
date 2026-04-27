import os
from dataclasses import asdict, dataclass
from typing import Any

import yaml

from src.features import get_feature_dim

EMNIST_BALANCED_CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
EMNIST_BYCLASS_CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
EMNIST_LETTERS_CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _parse_hidden_dims(hidden_dims: Any, hidden_dim: int) -> list[int]:
    if hidden_dims is None:
        return [int(hidden_dim)]
    if isinstance(hidden_dims, int):
        return [int(hidden_dims)]
    if isinstance(hidden_dims, str):
        items = [item.strip() for item in hidden_dims.split(",") if item.strip()]
        if not items:
            return [int(hidden_dim)]
        return [int(item) for item in items]
    if isinstance(hidden_dims, (list, tuple)):
        parsed = [int(item) for item in hidden_dims]
        if not parsed:
            return [int(hidden_dim)]
        return parsed
    raise TypeError("hidden_dims 必须是 int、list、tuple 或逗号分隔字符串。")


@dataclass
class BPTrainingHparams:
    dataset: str = "mnist"
    emnist_split: str = "balanced"
    data_root: str = "./data"
    feature_type: str = "pixel"

    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 2000

    hidden_dim: int = 128
    hidden_dims: list[int] | None = None
    num_classes: int = 10
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    weight_init: str = "kaiming"

    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0

    scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.5
    scheduler_patience: int = 3
    scheduler_monitor: str = "val_loss"

    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 0.0

    normalize_images: bool = False
    normalize_features: bool = False
    augment_train: bool = False
    augment_rotation: float = 10.0
    augment_translate: float = 0.1
    augment_scale_min: float = 0.95
    augment_scale_max: float = 1.05

    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 0

    early_stopping: bool = False
    patience: int = 10

    device: str = "cpu"
    seed: int = 42
    save_dir: str = "checkpoints"
    result_dir: str = "results"
    experiment_name: str = "bp_mnist"
    max_wrong_samples: int = 16

    def __post_init__(self):
        self.dataset = self.dataset.lower()
        self.emnist_split = self.emnist_split.lower()
        self.hidden_dims = _parse_hidden_dims(self.hidden_dims, self.hidden_dim)

        if self.dataset not in {"mnist", "emnist"}:
            raise ValueError("dataset 仅支持 'mnist' 或 'emnist'。")
        if self.dataset == "emnist" and self.emnist_split not in {"balanced", "byclass", "letters"}:
            raise ValueError("emnist_split 仅支持 'balanced'、'byclass'、'letters'。")
        if self.feature_type not in {"pixel", "pixel_projection", "pixel_projection_profile"}:
            raise ValueError(
                "feature_type 仅支持 'pixel'、'pixel_projection'、'pixel_projection_profile'。"
            )
        if self.optimizer not in {"sgd", "sgd_momentum", "adam", "adamw"}:
            raise ValueError("optimizer 仅支持 'sgd'、'sgd_momentum'、'adam'、'adamw'。")
        if self.scheduler not in {"none", "step", "reduce_on_plateau"}:
            raise ValueError("scheduler 仅支持 'none'、'step'、'reduce_on_plateau'。")
        if self.activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation 仅支持 'relu'、'sigmoid'、'tanh'。")
        if self.weight_init not in {"kaiming", "xavier"}:
            raise ValueError("weight_init 仅支持 'kaiming' 或 'xavier'。")
        if self.loss_name not in {"cross_entropy"}:
            raise ValueError("当前 loss_name 仅支持 'cross_entropy'。")
        if self.scheduler_monitor not in {"val_loss", "val_acc"}:
            raise ValueError("scheduler_monitor 仅支持 'val_loss' 或 'val_acc'。")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 取值范围应在 [0, 1) 之间。")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing 取值范围应在 [0, 1) 之间。")
        if self.augment_scale_min <= 0 or self.augment_scale_max <= 0:
            raise ValueError("augment_scale_min 和 augment_scale_max 必须大于 0。")
        if self.augment_scale_min > self.augment_scale_max:
            raise ValueError("augment_scale_min 不能大于 augment_scale_max。")
        if self.num_classes != len(self.resolved_class_names):
            raise ValueError(
                f"num_classes={self.num_classes} 与当前数据集类别数 "
                f"{len(self.resolved_class_names)} 不一致。"
            )

    @property
    def input_dim(self) -> int:
        return get_feature_dim(self.feature_type)

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.save_dir, f"{self.experiment_name}_best.pth")

    @property
    def experiment_result_dir(self) -> str:
        return os.path.join(self.result_dir, self.experiment_name)

    @property
    def resolved_class_names(self) -> list[str]:
        if self.dataset == "mnist":
            return [str(i) for i in range(10)]
        if self.emnist_split == "balanced":
            return EMNIST_BALANCED_CLASS_NAMES.copy()
        if self.emnist_split == "byclass":
            return EMNIST_BYCLASS_CLASS_NAMES.copy()
        if self.emnist_split == "letters":
            return EMNIST_LETTERS_CLASS_NAMES.copy()
        raise ValueError(f"未知 emnist_split: {self.emnist_split}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_hparams(cls, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
