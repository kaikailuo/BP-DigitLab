"""DataFrame helpers for the Streamlit app."""
from typing import Any, Dict, List

import pandas as pd


def format_experiment_history_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format experiment history rows for display."""
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


def format_probabilities_dataframe(
    probabilities: List[float],
    class_names: List[str] | None = None,
) -> pd.DataFrame:
    """Format class probabilities for display."""
    if class_names is None:
        class_names = [str(index) for index in range(len(probabilities))]
    if len(class_names) != len(probabilities):
        raise ValueError("class_names 数量必须与 probabilities 一致。")

    rows = [
        {"class": class_name, "probability": float(probability)}
        for class_name, probability in zip(class_names, probabilities)
    ]
    return pd.DataFrame(rows)
