import os

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.models import MLP
from src.utils import ensure_dir, plot_training_curves, resolve_device, save_json


def build_optimizer(model: nn.Module, config):
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.0,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"不支持的 optimizer: {config.optimizer}")


def build_scheduler(optimizer, config):
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    if config.scheduler == "reduce_on_plateau":
        mode = "min" if config.scheduler_monitor == "val_loss" else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=config.gamma,
            patience=config.scheduler_patience,
        )
    raise ValueError(f"不支持的 scheduler: {config.scheduler}")


def build_criterion(config):
    if config.loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    raise ValueError(f"不支持的 loss_name: {config.loss_name}")


class BPTrainer:
    def __init__(self, config, train_set, val_set):
        self.config = config
        self.train_set = train_set
        self.val_set = val_set

        self.device = resolve_device(config.device)
        self.model = MLP(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            activation=config.activation,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            weight_init=config.weight_init,
        ).to(self.device)

        self.criterion = build_criterion(config)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        self.train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        ensure_dir(config.save_dir)
        ensure_dir(config.experiment_result_dir)

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _run_one_epoch(self, loader, training: bool):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if training:
                    loss.backward()
                    if self.config.gradient_clip_norm > 0:
                        clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _save_checkpoint(self, epoch: int, best_val_acc: float):
        checkpoint = {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, self.config.checkpoint_path)

    def _save_history(self, best_epoch: int, best_val_acc: float):
        history_payload = dict(self.history)
        history_payload["best_epoch"] = best_epoch
        history_payload["best_val_acc"] = round(best_val_acc, 6)
        save_json(
            history_payload,
            os.path.join(self.config.experiment_result_dir, "training_history.json"),
        )
        if hasattr(self.train_set, "stats_summary"):
            save_json(
                self.train_set.stats_summary,
                os.path.join(self.config.experiment_result_dir, "data_stats.json"),
            )
        plot_training_curves(self.history, self.config.experiment_result_dir)

    def _step_scheduler(self, val_loss: float, val_acc: float):
        if self.scheduler is None:
            return
        if self.config.scheduler == "reduce_on_plateau":
            monitor_value = val_loss if self.config.scheduler_monitor == "val_loss" else val_acc
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def run(self):
        best_val_acc = -1.0
        best_epoch = 0
        patience_counter = 0

        print(f"开始训练实验：{self.config.experiment_name}")
        print(f"设备：{self.device}")
        print(f"输入维度：{self.config.input_dim}")
        print(f"隐藏层结构：{self.config.hidden_dims}")

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._run_one_epoch(self.train_loader, training=True)
            val_loss, val_acc = self._run_one_epoch(self.val_loader, training=False)

            self._step_scheduler(val_loss=val_loss, val_acc=val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 6))
            self.history["train_acc"].append(round(train_acc, 6))
            self.history["val_loss"].append(round(val_loss, 6))
            self.history["val_acc"].append(round(val_acc, 6))
            self.history["lr"].append(round(current_lr, 8))

            print(
                f"Epoch [{epoch:02d}/{self.config.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={current_lr:.6f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_acc)
                print(f"验证集准确率提升，已保存最佳模型到：{self.config.checkpoint_path}")
            else:
                patience_counter += 1

            if self.config.early_stopping and patience_counter >= self.config.patience:
                print(
                    f"连续 {self.config.patience} 个 epoch 验证集准确率未提升，触发 Early Stopping。"
                )
                break

        self._save_history(best_epoch=best_epoch, best_val_acc=best_val_acc)
        print(f"训练结束，最佳验证集准确率：{best_val_acc:.4f}，最佳轮次：{best_epoch}")
