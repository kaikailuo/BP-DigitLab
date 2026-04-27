"""Training page composition."""
from pathlib import Path
from typing import Dict, List

import streamlit as st
import torch

from app.components.experiment_history import render_experiment_history
from app.components.train_sidebar import render_train_sidebar
from app.services.form_service import build_hparams_from_form
from app.services.predictor_service import get_cached_predictor
from app.services.training_service import get_training_summary, run_training
from app.state import (
    set_loaded_checkpoint,
    set_loaded_config,
    set_loaded_device,
    set_ui_page,
    set_ui_result_dir,
    set_ui_save_dir,
)
from src.utils import set_seed


def render_train_page(workspace_root: Path) -> None:
    """Render the training page."""
    st.title("BP 手写字符识别实验 - 训练模型页面")
    st.caption("配置训练、查看历史实验，并从历史模型进入交互识别页面。")

    form_values = render_train_sidebar(workspace_root)

    st.subheader("训练状态 / 日志 / 指标")
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    if form_values.get("start_training"):
        try:
            config = build_hparams_from_form(form_values)
        except Exception as exc:
            status_placeholder.error(f"参数解析失败: {exc}")
            return

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            status_placeholder.warning(
                f"当前选择了 {config.device}，但本机未检测到可用 CUDA，训练会自动回退到 CPU。"
            )

        logs: List[str] = []
        progress = progress_placeholder.progress(0.0)

        def on_log(message: str) -> None:
            logs.append(message)
            logs_placeholder.code("\n".join(logs[-300:]), language="text")

        def on_epoch(metrics: Dict[str, float]) -> None:
            progress.progress(min(float(metrics["epoch"]) / float(metrics["total_epochs"]), 1.0))
            metrics_placeholder.table(
                [
                    {
                        "epoch": f"{int(metrics['epoch'])}/{int(metrics['total_epochs'])}",
                        "train_loss": round(float(metrics["train_loss"]), 6),
                        "train_acc": round(float(metrics["train_acc"]), 6),
                        "val_loss": round(float(metrics["val_loss"]), 6),
                        "val_acc": round(float(metrics["val_acc"]), 6),
                        "lr": round(float(metrics["lr"]), 8),
                    }
                ]
            )

        try:
            set_seed(config.seed)
            trainer = run_training(config, log_callback=on_log, epoch_callback=on_epoch)

            summary = get_training_summary(trainer)
            checkpoint_path = str(Path(config.checkpoint_path).resolve())

            set_loaded_checkpoint(checkpoint_path)
            set_loaded_device(config.device)
            set_loaded_config(config.to_dict())
            set_ui_save_dir(form_values["save_dir"])
            set_ui_result_dir(form_values["result_dir"])

            status_placeholder.success(
                f"训练完成: {config.experiment_name} | "
                f"best_val_acc={summary['best_val_acc']:.4f} | "
                f"checkpoint={checkpoint_path}"
            )
        except Exception as exc:
            status_placeholder.error(f"训练失败: {exc}")

    st.divider()

    def on_enter_recognition(selected_row: Dict[str, str], form_data: Dict[str, str]) -> None:
        checkpoint_path = selected_row["checkpoint_path"]
        device = form_data["device"]

        try:
            predictor = get_cached_predictor(checkpoint_path, device)
            set_loaded_checkpoint(checkpoint_path)
            set_loaded_device(device)
            set_loaded_config(predictor.config.to_dict())
            set_ui_page("interactive")
            st.switch_page("pages/2_interactive_recognition.py")
        except Exception as exc:
            st.error(f"模型加载校验失败，无法进入识别页: {exc}")

    render_experiment_history(
        form_values,
        on_enter_recognition=on_enter_recognition,
    )
