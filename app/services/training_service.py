"""
训练服务。

负责训练流程的编排，包括训练启动、日志回调、指标回调等。
"""
from typing import Any, Callable, Dict, List, Optional

import torch

from src.datasets import build_train_val_datasets
from src.trainer import BPTrainer
from src.utils import set_seed
from src.hparams import BPTrainingHparams


def run_training(
    config: BPTrainingHparams,
    log_callback: Optional[Callable[[str], None]] = None,
    epoch_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> BPTrainer:
    """
    运行一次完整的训练。
    
    参数：
        config: 训练配置
        log_callback: 日志回调函数，接收日志字符串
        epoch_callback: epoch 回调函数，接收指标字典
    
    返回：
        已完成训练的 BPTrainer 实例
    
    异常：
        ValueError: 参数验证失败
        RuntimeError: 训练过程中发生错误
    """
    # 设置随机种子
    set_seed(config.seed)
    
    # 检查 CUDA 可用性
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        if log_callback:
            log_callback(f"警告：选择了 {config.device}，但本机无可用 CUDA，自动退回到 CPU")
    
    # 构建数据集
    train_set, val_set = build_train_val_datasets(config)
    
    # 创建训练器
    trainer = BPTrainer(
        config=config,
        train_set=train_set,
        val_set=val_set,
        log_callback=log_callback,
        epoch_callback=epoch_callback,
    )
    
    # 运行训练
    trainer.run()
    
    return trainer


def get_training_summary(trainer: BPTrainer) -> Dict[str, Any]:
    """
    从训练器获取训练摘要。
    
    参数：
        trainer: 已完成的 BPTrainer 实例
    
    返回：
        包含关键指标的摘要字典
    """
    history = trainer.history or {}
    val_acc_list = history.get("val_acc", []) or [0.0]
    best_val_acc = max(val_acc_list)
    
    return {
        "best_val_acc": best_val_acc,
        "total_epochs": len(val_acc_list),
        "training_history": history,
    }
