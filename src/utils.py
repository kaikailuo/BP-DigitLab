import json
import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("配置中请求使用 CUDA，但当前环境不可用，已自动切换到 CPU。")
        return torch.device("cpu")
    return torch.device(device_name)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _to_serializable(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_to_serializable)


def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_training_curves(history: Dict[str, List[float]], save_dir: str):
    ensure_dir(save_dir)
    epochs = history["epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 10))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(epochs, history["train_loss"], label="train_loss")
    ax1.plot(epochs, history["val_loss"], label="val_loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(epochs, history["train_acc"], label="train_acc")
    ax2.plot(epochs, history["val_acc"], label="val_acc")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(epochs, history["lr"], label="learning_rate")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_curve.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], save_path: str):
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size > 0 else 0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_wrong_samples(wrong_samples: List[Dict], save_path: str, max_items: int = 16):
    ensure_dir(os.path.dirname(save_path))
    if not wrong_samples:
        print("测试集中没有错误样本，跳过错误样本可视化。")
        return

    display_samples = wrong_samples[:max_items]
    cols = 4
    rows = int(np.ceil(len(display_samples) / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, sample in enumerate(display_samples, start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(sample["image"].squeeze(0), cmap="gray")
        plt.title(f"T:{sample['true']} / P:{sample['pred']}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
