"""
训练页面编排。

组合训练参数侧边栏、训练状态显示和历史实验管理。
"""
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import torch

from app.components.train_sidebar import render_train_sidebar
from app.components.experiment_history import render_experiment_history
from app.services.form_service import build_hparams_from_form
from app.services.training_service import run_training, get_training_summary
from app.services.model_service import get_cached_model_bundle
from app.state import set_ui_page, set_loaded_checkpoint, set_loaded_device, set_loaded_config, set_ui_save_dir, set_ui_result_dir
from src.utils import set_seed


def render_train_page(workspace_root: Path) -> None:
    """
    渲染训练页面主体。
    
    参数：
        workspace_root: 工作空间根目录
    """
    st.title("BP 手写数字识别实验 - 训练模型页面")
    st.caption("课程实验演示版：配置训练 -> 管理历史实验 -> 加载历史最佳模型")

    # 1. 侧边栏表单
    form_values = render_train_sidebar(workspace_root)
    
    # 2. 训练区域
    st.subheader("训练状态 / 日志 / 指标")
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    if form_values.get("start_training"):
        try:
            # 解析并验证参数
            config = build_hparams_from_form(form_values)
        except Exception as exc:
            status_placeholder.error(f"参数解析失败: {exc}")
            return

        # 检查 CUDA 可用性
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            status_placeholder.warning(
                f"当前选择了 {config.device}，但本机 PyTorch 没检测到可用 CUDA，训练会自动退回到 CPU。"
            )

        logs: List[str] = []
        progress = progress_placeholder.progress(0.0)

        def on_log(message: str):
            logs.append(message)
            logs_placeholder.code("\n".join(logs[-300:]), language="text")

        def on_epoch(metrics: Dict[str, float]):
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
            
            # 更新会话状态以便识别页使用
            set_loaded_checkpoint(checkpoint_path)
            set_loaded_device(config.device)
            set_loaded_config(config.to_dict())
            set_ui_save_dir(form_values["save_dir"])
            set_ui_result_dir(form_values["result_dir"])

            status_placeholder.success(
                f"训练完成: {config.experiment_name} | best_val_acc={summary['best_val_acc']:.4f} | checkpoint={checkpoint_path}"
            )
        except Exception as exc:
            status_placeholder.error(f"训练失败: {exc}")

    st.divider()

    # 3. 历史实验管理
    def on_load_model(selected_row: Dict[str, Any], form_data: Dict[str, Any]) -> None:
        """加载选中的模型。"""
        try:
            _, config, _ = get_cached_model_bundle(
                checkpoint_path=selected_row["checkpoint_path"],
                device=form_data["device"],
            )
            set_loaded_checkpoint(selected_row["checkpoint_path"])
            set_loaded_device(form_data["device"])
            set_loaded_config(config.to_dict())
            st.success("模型已加载，可进入交互识别页面。")
        except Exception as exc:
            st.error(f"模型加载失败: {exc}")

    def on_enter_recognition(selected_row: Dict[str, Any], form_data: Dict[str, Any]) -> None:
        """进入识别页。"""
        set_loaded_checkpoint(selected_row["checkpoint_path"])
        set_loaded_device(form_data["device"])
        set_ui_page("interactive")
        st.switch_page("pages/2_interactive_recognition.py")

    render_experiment_history(
        form_values,
        on_load_model=on_load_model,
        on_enter_recognition=on_enter_recognition,
    )
