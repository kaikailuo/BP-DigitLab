from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml
import torch

from src.datasets import build_train_val_datasets
from experiment_manager import (
    build_hparams_from_form,
    load_experiment_artifacts,
    parse_hidden_dims,
    scan_experiments,
)
from src.hparams import BPTrainingHparams
from inference import load_model_from_checkpoint
from src.trainer import BPTrainer
from src.utils import set_seed

APP_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent


def _normalize_device_choice(value: Any) -> str:
    return "cpu" if str(value).strip() == "cpu" else "cuda:0"
st.set_page_config(page_title="BP DigitLab - 训练模型", layout="wide")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] {
        font-size: 1.18rem !important;
        font-weight: 700;
        min-height: 2.7rem;
        padding: 0.8rem 0.9rem !important;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded if isinstance(loaded, dict) else {}


def _format_hidden_dims_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = text
        if isinstance(parsed, (list, tuple)):
            return ",".join(str(item).strip() for item in parsed if str(item).strip())
    return text


def default_form_values() -> Dict[str, Any]:
    base = BPTrainingHparams().to_dict()
    experiment_name = base["experiment_name"]
    save_dir = str((WORKSPACE_ROOT / base["save_dir"]).resolve())

    return {
        "experiment_name": experiment_name,
        "epochs": base["epochs"],
        "batch_size": base["batch_size"],
        "learning_rate": base["lr"],
        "hidden_dims": str(base["hidden_dim"]),
        "dropout": base.get("dropout", 0.0),
        "optimizer": base["optimizer"],
        "seed": base["seed"],
        "feature_type": base["feature_type"],
        "checkpoint_path": str(Path(save_dir) / f"{experiment_name}_best.pth"),
        "auto_checkpoint_path": True,
        "train_size": base["train_size"],
        "val_size": base["val_size"],
        "test_size": base["test_size"],
        "momentum": base["momentum"],
        "weight_decay": base["weight_decay"],
        "scheduler": base["scheduler"],
        "step_size": base["step_size"],
        "gamma": base["gamma"],
        "early_stopping": base["early_stopping"],
        "patience": base["patience"],
        "device": _normalize_device_choice(base["device"]),
        "data_root": str((WORKSPACE_ROOT / base["data_root"]).resolve()),
        "save_dir": save_dir,
        "result_dir": str((WORKSPACE_ROOT / base["result_dir"]).resolve()),
        "num_workers": base["num_workers"],
    }


def apply_values_to_state(values: Dict[str, Any]):
    for key, value in values.items():
        st.session_state[f"cfg_{key}"] = value


def ensure_form_state_initialized():
    defaults = default_form_values()

    # In multi-page navigation or after config schema changes, the init flag may
    # exist while some cfg_* keys are missing. Always backfill missing keys.
    for key, value in defaults.items():
        session_key = f"cfg_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = value

    st.session_state["cfg_initialized"] = True


def load_preset_into_form(yaml_path: Path):
    loaded = load_yaml_file(yaml_path)
    defaults = default_form_values()

    merged = dict(defaults)
    merged.update(
        {
            "experiment_name": loaded.get("experiment_name", defaults["experiment_name"]),
            "epochs": loaded.get("epochs", defaults["epochs"]),
            "batch_size": loaded.get("batch_size", defaults["batch_size"]),
            "learning_rate": loaded.get("lr", defaults["learning_rate"]),
            "hidden_dims": _format_hidden_dims_value(loaded.get("hidden_dims", loaded.get("hidden_dim", defaults["hidden_dims"]))),
            "dropout": loaded.get("dropout", defaults["dropout"]),
            "optimizer": loaded.get("optimizer", defaults["optimizer"]),
            "seed": loaded.get("seed", defaults["seed"]),
            "feature_type": loaded.get("feature_type", defaults["feature_type"]),
            "train_size": loaded.get("train_size", defaults["train_size"]),
            "val_size": loaded.get("val_size", defaults["val_size"]),
            "test_size": loaded.get("test_size", defaults["test_size"]),
            "momentum": loaded.get("momentum", defaults["momentum"]),
            "weight_decay": loaded.get("weight_decay", defaults["weight_decay"]),
            "scheduler": loaded.get("scheduler", defaults["scheduler"]),
            "step_size": loaded.get("step_size", defaults["step_size"]),
            "gamma": loaded.get("gamma", defaults["gamma"]),
            "early_stopping": loaded.get("early_stopping", defaults["early_stopping"]),
            "patience": loaded.get("patience", defaults["patience"]),
            "device": _normalize_device_choice(loaded.get("device", defaults["device"])),
            "data_root": str((WORKSPACE_ROOT / loaded.get("data_root", "./data")).resolve()),
            "save_dir": str((WORKSPACE_ROOT / loaded.get("save_dir", "checkpoints")).resolve()),
            "result_dir": str((WORKSPACE_ROOT / loaded.get("result_dir", "results")).resolve()),
            "num_workers": loaded.get("num_workers", defaults["num_workers"]),
        }
    )

    save_dir = Path(str(merged["save_dir"]))
    merged["checkpoint_path"] = str(save_dir / f"{merged['experiment_name']}_best.pth")
    merged["auto_checkpoint_path"] = True
    apply_values_to_state(merged)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(checkpoint_path: str, device: str):
    return load_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)


@st.cache_resource(show_spinner=False)
def _load_interactive_page_module():
    module_path = APP_ROOT / "2_interactive_recognition.py"
    spec = importlib.util.spec_from_file_location("bp_digitlab_interactive_page", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载交互页模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("训练参数")

    preset_dir = WORKSPACE_ROOT / "hparams"
    preset_files = sorted(preset_dir.glob("*.yaml")) if preset_dir.exists() else []
    preset_names = [p.name for p in preset_files]

    selected_preset = st.sidebar.selectbox("预设配置", options=["(不使用预设)"] + preset_names)
    if selected_preset != "(不使用预设)" and st.sidebar.button("应用预设到表单"):
        load_preset_into_form(preset_dir / selected_preset)

    experiment_name = st.sidebar.text_input("experiment_name", key="cfg_experiment_name")
    epochs = st.sidebar.number_input("epochs", min_value=1, value=int(st.session_state["cfg_epochs"]), key="cfg_epochs")
    batch_size = st.sidebar.number_input("batch_size", min_value=1, value=int(st.session_state["cfg_batch_size"]), key="cfg_batch_size")
    learning_rate = st.sidebar.number_input(
        "learning_rate",
        min_value=1e-6,
        value=float(st.session_state["cfg_learning_rate"]),
        format="%.6f",
        key="cfg_learning_rate",
    )
    hidden_dims = st.sidebar.text_input("hidden_dims (e.g. 256,128)", key="cfg_hidden_dims")
    dropout = st.sidebar.number_input(
        "dropout", min_value=0.0, max_value=0.99, value=float(st.session_state["cfg_dropout"]), key="cfg_dropout"
    )
    optimizer = st.sidebar.selectbox(
        "optimizer",
        options=["sgd", "sgd_momentum", "adam"],
        index=["sgd", "sgd_momentum", "adam"].index(st.session_state["cfg_optimizer"]),
        key="cfg_optimizer",
    )
    seed = st.sidebar.number_input("seed", min_value=0, value=int(st.session_state["cfg_seed"]), key="cfg_seed")

    feature_type = st.sidebar.selectbox(
        "feature_type",
        options=["pixel", "pixel_projection"],
        index=["pixel", "pixel_projection"].index(st.session_state["cfg_feature_type"]),
        key="cfg_feature_type",
    )

    device = st.sidebar.selectbox(
        "device",
        options=["cpu", "cuda:0"],
        index=0 if _normalize_device_choice(st.session_state["cfg_device"]) == "cpu" else 1,
        key="cfg_device",
    )

    with st.sidebar.expander("高级参数", expanded=False):
        train_size = st.number_input("train_size", min_value=1, value=int(st.session_state["cfg_train_size"]), key="cfg_train_size")
        val_size = st.number_input("val_size", min_value=1, value=int(st.session_state["cfg_val_size"]), key="cfg_val_size")
        test_size = st.number_input("test_size", min_value=1, value=int(st.session_state["cfg_test_size"]), key="cfg_test_size")
        momentum = st.number_input("momentum", min_value=0.0, max_value=0.999, value=float(st.session_state["cfg_momentum"]), key="cfg_momentum")
        weight_decay = st.number_input(
            "weight_decay",
            min_value=0.0,
            value=float(st.session_state["cfg_weight_decay"]),
            format="%.6f",
            key="cfg_weight_decay",
        )
        scheduler = st.selectbox(
            "scheduler",
            options=["none", "step"],
            index=["none", "step"].index(st.session_state["cfg_scheduler"]),
            key="cfg_scheduler",
        )
        step_size = st.number_input("step_size", min_value=1, value=int(st.session_state["cfg_step_size"]), key="cfg_step_size")
        gamma = st.number_input("gamma", min_value=0.01, max_value=1.0, value=float(st.session_state["cfg_gamma"]), key="cfg_gamma")
        early_stopping = st.checkbox("early_stopping", value=bool(st.session_state["cfg_early_stopping"]), key="cfg_early_stopping")
        patience = st.number_input("patience", min_value=1, value=int(st.session_state["cfg_patience"]), key="cfg_patience")
        data_root = st.text_input("data_root", key="cfg_data_root")
        save_dir = st.text_input("save_dir", key="cfg_save_dir")
        result_dir = st.text_input("result_dir", key="cfg_result_dir")
        num_workers = st.number_input("num_workers", min_value=0, value=int(st.session_state["cfg_num_workers"]), key="cfg_num_workers")

    auto_checkpoint_path = st.sidebar.checkbox(
        "checkpoint_path 自动生成",
        value=bool(st.session_state["cfg_auto_checkpoint_path"]),
        key="cfg_auto_checkpoint_path",
    )

    auto_path = str(Path(str(st.session_state["cfg_save_dir"])) / f"{experiment_name}_best.pth")
    if auto_checkpoint_path:
        checkpoint_path = auto_path
        st.sidebar.code(checkpoint_path)
    else:
        checkpoint_path = st.sidebar.text_input("checkpoint_path", key="cfg_checkpoint_path")

    start_training = st.sidebar.button("开始训练", type="primary")

    try:
        parsed_hidden_dims = parse_hidden_dims(hidden_dims)
        if len(parsed_hidden_dims) > 1:
            st.sidebar.info(f"当前模型是单隐藏层，将使用 hidden_dim={parsed_hidden_dims[0]}。")
    except Exception:
        st.sidebar.warning("hidden_dims 格式错误，训练前请修正。")

    if dropout > 0:
        st.sidebar.caption("当前项目模型未使用 dropout，该值仅记录在配置中。")

    form = {
        "experiment_name": experiment_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "optimizer": optimizer,
        "seed": seed,
        "feature_type": feature_type,
        "checkpoint_path": checkpoint_path,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "step_size": step_size,
        "gamma": gamma,
        "early_stopping": early_stopping,
        "patience": patience,
        "device": device,
        "data_root": data_root,
        "save_dir": save_dir,
        "result_dir": result_dir,
        "num_workers": num_workers,
        "start_training": start_training,
    }

    st.session_state["ui_save_dir"] = save_dir
    st.session_state["ui_result_dir"] = result_dir
    return form


def render_training_area(form_data: Dict[str, Any]):
    st.subheader("训练状态 / 日志 / 指标")
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()

    if not form_data["start_training"]:
        status_placeholder.info("等待开始训练。")
        return

    try:
        config = build_hparams_from_form(form_data)
    except Exception as exc:
        status_placeholder.error(f"参数解析失败: {exc}")
        return

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
        train_set, val_set = build_train_val_datasets(config)
        trainer = BPTrainer(
            config=config,
            train_set=train_set,
            val_set=val_set,
            log_callback=on_log,
            epoch_callback=on_epoch,
        )
        trainer.run()

        best_val_acc = max(trainer.history.get("val_acc", []) or [0.0])
        checkpoint_path = str(Path(config.checkpoint_path).resolve())
        st.session_state["loaded_checkpoint"] = checkpoint_path
        st.session_state["loaded_device"] = config.device
        st.session_state["loaded_config"] = config.to_dict()

        status_placeholder.success(
            f"训练完成: {config.experiment_name} | best_val_acc={best_val_acc:.4f} | checkpoint={checkpoint_path}"
        )
    except Exception as exc:
        status_placeholder.error(f"训练失败: {exc}")


def _history_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    data = []
    for row in rows:
        data.append(
            {
                "experiment_name": row["experiment_name"],
                "checkpoint_file": row["checkpoint_file"],
                "best_val_acc": row["best_val_acc"],
                "training_history": row["has_training_history"],
                "loss_curve": row["has_loss_curve"],
                "accuracy_curve": row["has_accuracy_curve"],
                "test_metrics": row["has_test_metrics"],
            }
        )
    return pd.DataFrame(data)


def render_experiment_history(form_data: Dict[str, Any]):
    st.subheader("历史实验记录")

    save_dir = Path(str(form_data["save_dir"]))
    result_dir = Path(str(form_data["result_dir"]))

    if not save_dir.exists():
        st.warning(f"checkpoints 目录不存在: {save_dir}")
    if not result_dir.exists():
        st.warning(f"results 目录不存在: {result_dir}")

    rows = scan_experiments(
        checkpoint_root=str(save_dir),
        result_root=str(result_dir),
    )

    if not rows:
        st.info("未扫描到历史实验，请先训练至少一次。")
        return

    st.dataframe(_history_dataframe(rows), width="stretch", hide_index=True)

    selected_index = st.selectbox(
        "选择历史实验 / 历史模型 checkpoint",
        options=list(range(len(rows))),
        format_func=lambda i: f"{rows[i]['experiment_name']} | {rows[i]['checkpoint_file']}",
    )
    selected_row = rows[selected_index]

    st.caption(f"当前选择: {selected_row['experiment_name']}")

    if selected_row["checkpoint_path"]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("加载该模型进行识别（无需重新训练）"):
                try:
                    _, config, _ = cached_model_bundle(
                        checkpoint_path=selected_row["checkpoint_path"],
                        device=form_data["device"],
                    )
                    st.session_state["loaded_checkpoint"] = selected_row["checkpoint_path"]
                    st.session_state["loaded_device"] = form_data["device"]
                    st.session_state["loaded_config"] = config.to_dict()
                    st.success("模型已加载，可进入交互识别页面。")
                except Exception as exc:
                    st.error(f"模型加载失败: {exc}")
        with col2:
            if st.button("进入交互识别页面"):
                st.session_state["loaded_checkpoint"] = selected_row["checkpoint_path"]
                st.session_state["loaded_device"] = form_data["device"]
                st.session_state["ui_page"] = "interactive"
                st.rerun()
    else:
        st.info("该实验只有 results，没有可加载的 checkpoint。")

    artifacts = load_experiment_artifacts(selected_row)

    st.markdown("#### 实验结果可视化")
    image_paths = artifacts.get("images", {})
    if image_paths:
        for image_name, image_path in image_paths.items():
            st.image(image_path, caption=image_name, width="stretch")
    else:
        st.info("未找到可视化图片文件（loss/accuracy/confusion 等）。")

    metrics = artifacts.get("test_metrics", {})
    if metrics:
        st.markdown("#### test_metrics.json")
        st.json(metrics)

    report_text = artifacts.get("classification_report", "")
    if report_text:
        st.markdown("#### classification_report.txt")
        st.code(report_text, language="text")


ensure_form_state_initialized()
st.session_state.setdefault("ui_page", "train")

if st.session_state["ui_page"] == "interactive":
    interactive_module = _load_interactive_page_module()
    interactive_module.render_interactive_page()
    st.stop()

st.title("BP 手写数字识别实验 - 训练模型页面")
st.caption("课程实验演示版：配置训练 -> 管理历史实验 -> 加载历史最佳模型")

form_values = render_sidebar()
render_training_area(form_values)
st.divider()
render_experiment_history(form_values)