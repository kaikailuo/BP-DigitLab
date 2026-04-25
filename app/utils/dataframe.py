"""
数据框工具。

提供数据框格式化和处理辅助函数。
"""
from typing import Any, Dict, List

import pandas as pd


def format_experiment_history_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    格式化实验历史列表为 DataFrame。
    
    参数：
        rows: 从 experiment_service.scan_experiments 获得的行列表
    
    返回：
        格式化后的 DataFrame
    """
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
                "metrics": row["has_metrics"],
            }
        )
    return pd.DataFrame(data)


def format_probabilities_dataframe(probabilities: List[float]) -> pd.DataFrame:
    """
    格式化预测概率为 DataFrame。
    
    参数：
        probabilities: 预测概率列表 (10 个值，对应数字 0-9)
    
    返回：
        包含数字和概率的 DataFrame
    """
    probability_rows = [
        {"digit": digit, "probability": float(prob)}
        for digit, prob in enumerate(probabilities)
    ]
    return pd.DataFrame(probability_rows)
