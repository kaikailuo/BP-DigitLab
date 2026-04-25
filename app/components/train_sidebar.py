"""
训练参数侧边栏组件。

负责渲染训练参数表单。
"""
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app.services.preset_service import get_preset_list, load_preset
from app.services.form_service import (
    parse_hidden_dims,
    format_hidden_dims_value,
    get_default_form_values,
)
from app.state import update_form_values, get_form_value


def render_train_sidebar(workspace_root: Path) -> Dict[str, Any]:
    """
    渲染训练参数侧边栏。
    
    参数：
        workspace_root: 工作空间根目录
    
    返回：
        包含所有表单值的字典
    """
    st.sidebar.header("训练参数")

    # 预设配置加载
    preset_dir = workspace_root / "hparams"
    preset_names = get_preset_list(preset_dir)
    
    selected_preset = st.sidebar.selectbox(
        "预设配置",
        options=["(不使用预设)"] + preset_names
    )
    
    if selected_preset != "(不使用预设)" and st.sidebar.button("应用预设到表单"):
        preset_data = load_preset(preset_dir / selected_preset)
        defaults = get_default_form_values(workspace_root)
        merged = dict(defaults)
        
        merged.update({
            "experiment_name": preset_data.get("experiment_name", defaults["experiment_name"]),
            "epochs": preset_data.get("epochs", defaults["epochs"]),
            "batch_size": preset_data.get("batch_size", defaults["batch_size"]),
            "learning_rate": preset_data.get("lr", defaults["learning_rate"]),
            "hidden_dims": format_hidden_dims_value(
                preset_data.get("hidden_dims", preset_data.get("hidden_dim", defaults["hidden_dims"]))
            ),
            "dropout": preset_data.get("dropout", defaults["dropout"]),
            "optimizer": preset_data.get("optimizer", defaults["optimizer"]),
            "seed": preset_data.get("seed", defaults["seed"]),
            "feature_type": preset_data.get("feature_type", defaults["feature_type"]),
            "train_size": preset_data.get("train_size", defaults["train_size"]),
            "val_size": preset_data.get("val_size", defaults["val_size"]),
            "test_size": preset_data.get("test_size", defaults["test_size"]),
            "momentum": preset_data.get("momentum", defaults["momentum"]),
            "weight_decay": preset_data.get("weight_decay", defaults["weight_decay"]),
            "scheduler": preset_data.get("scheduler", defaults["scheduler"]),
            "step_size": preset_data.get("step_size", defaults["step_size"]),
            "gamma": preset_data.get("gamma", defaults["gamma"]),
            "early_stopping": preset_data.get("early_stopping", defaults["early_stopping"]),
            "patience": preset_data.get("patience", defaults["patience"]),
            "device": preset_data.get("device", defaults["device"]),
            "data_root": str((workspace_root / preset_data.get("data_root", "./data")).resolve()),
            "save_dir": str((workspace_root / preset_data.get("save_dir", "checkpoints")).resolve()),
            "result_dir": str((workspace_root / preset_data.get("result_dir", "results")).resolve()),
            "num_workers": preset_data.get("num_workers", defaults["num_workers"]),
        })
        
        save_dir = Path(str(merged["save_dir"]))
        merged["checkpoint_path"] = str(save_dir / f"{merged['experiment_name']}_best.pth")
        merged["auto_checkpoint_path"] = True
        
        update_form_values(merged)

    # 主参数表单
    experiment_name = st.sidebar.text_input("experiment_name", key="cfg_experiment_name")
    epochs = st.sidebar.number_input("epochs", min_value=1, value=int(get_form_value("epochs", 10)), key="cfg_epochs")
    batch_size = st.sidebar.number_input("batch_size", min_value=1, value=int(get_form_value("batch_size", 64)), key="cfg_batch_size")
    learning_rate = st.sidebar.number_input(
        "learning_rate",
        min_value=1e-6,
        value=float(get_form_value("learning_rate", 0.001)),
        format="%.6f",
        key="cfg_learning_rate",
    )
    hidden_dims = st.sidebar.text_input("hidden_dims (e.g. 256,128)", key="cfg_hidden_dims")
    dropout = st.sidebar.number_input(
        "dropout",
        min_value=0.0,
        max_value=0.99,
        value=float(get_form_value("dropout", 0.0)),
        key="cfg_dropout"
    )
    optimizer = st.sidebar.selectbox(
        "optimizer",
        options=["sgd", "sgd_momentum", "adam"],
        index=["sgd", "sgd_momentum", "adam"].index(get_form_value("optimizer", "adam")),
        key="cfg_optimizer",
    )
    seed = st.sidebar.number_input("seed", min_value=0, value=int(get_form_value("seed", 42)), key="cfg_seed")
    feature_type = st.sidebar.selectbox(
        "feature_type",
        options=["pixel", "pixel_projection"],
        index=["pixel", "pixel_projection"].index(get_form_value("feature_type", "pixel")),
        key="cfg_feature_type",
    )
    device = st.sidebar.selectbox(
        "device",
        options=["cpu", "cuda:0"],
        index=0 if str(get_form_value("device", "cpu")).strip() == "cpu" else 1,
        key="cfg_device",
    )

    # 高级参数
    with st.sidebar.expander("高级参数", expanded=False):
        train_size = st.number_input("train_size", min_value=1, value=int(get_form_value("train_size", 50000)), key="cfg_train_size")
        val_size = st.number_input("val_size", min_value=1, value=int(get_form_value("val_size", 10000)), key="cfg_val_size")
        test_size = st.number_input("test_size", min_value=1, value=int(get_form_value("test_size", 10000)), key="cfg_test_size")
        momentum = st.number_input("momentum", min_value=0.0, max_value=0.999, value=float(get_form_value("momentum", 0.9)), key="cfg_momentum")
        weight_decay = st.number_input(
            "weight_decay",
            min_value=0.0,
            value=float(get_form_value("weight_decay", 0.0)),
            format="%.6f",
            key="cfg_weight_decay",
        )
        scheduler = st.selectbox(
            "scheduler",
            options=["none", "step" , "reduce_on_plateau"],
            index=["none", "step", "reduce_on_plateau"].index(get_form_value("scheduler", "none")),
            key="cfg_scheduler",
        )
        step_size = st.number_input("step_size", min_value=1, value=int(get_form_value("step_size", 10)), key="cfg_step_size")
        gamma = st.number_input("gamma", min_value=0.01, max_value=1.0, value=float(get_form_value("gamma", 0.1)), key="cfg_gamma")
        early_stopping = st.checkbox("early_stopping", value=bool(get_form_value("early_stopping", False)), key="cfg_early_stopping")
        patience = st.number_input("patience", min_value=1, value=int(get_form_value("patience", 5)), key="cfg_patience")
        data_root = st.text_input("data_root", key="cfg_data_root")
        save_dir = st.text_input("save_dir", key="cfg_save_dir")
        result_dir = st.text_input("result_dir", key="cfg_result_dir")
        num_workers = st.number_input("num_workers", min_value=0, value=int(get_form_value("num_workers", 0)), key="cfg_num_workers")

    # Checkpoint 路径管理
    auto_checkpoint_path = st.sidebar.checkbox(
        "checkpoint_path 自动生成",
        value=bool(get_form_value("auto_checkpoint_path", True)),
        key="cfg_auto_checkpoint_path",
    )

    auto_path = str(Path(str(get_form_value("save_dir", "checkpoints"))) / f"{experiment_name}_best.pth")
    if auto_checkpoint_path:
        checkpoint_path = auto_path
        st.sidebar.code(checkpoint_path)
    else:
        checkpoint_path = st.sidebar.text_input("checkpoint_path", key="cfg_checkpoint_path")

    start_training = st.sidebar.button("开始训练", type="primary")

    # 参数验证提示
    try:
        parsed_hidden_dims = parse_hidden_dims(hidden_dims)
        if len(parsed_hidden_dims) > 1:
            st.sidebar.info(f"当前模型是单隐藏层，将使用 hidden_dim={parsed_hidden_dims[0]}。")
    except Exception:
        st.sidebar.warning("hidden_dims 格式错误，训练前请修正。")

    if dropout > 0:
        st.sidebar.caption("当前项目模型未使用 dropout，该值仅记录在配置中。")

    return {
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
