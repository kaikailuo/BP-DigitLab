"""
应用层数据结构定义。

使用 TypedDict 和其他类型工具，为跨模块数据交互提供类型提示。
"""
from typing import Any, Dict, List, Optional, TypedDict


class FormData(TypedDict, total=False):
    """训练参数表单数据。"""
    experiment_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dims: str
    dropout: float
    optimizer: str
    seed: int
    feature_type: str
    checkpoint_path: str
    auto_checkpoint_path: bool
    train_size: int
    val_size: int
    test_size: int
    momentum: float
    weight_decay: float
    scheduler: str
    step_size: int
    gamma: float
    early_stopping: bool
    patience: int
    device: str
    data_root: str
    save_dir: str
    result_dir: str
    num_workers: int
    start_training: bool


class TrainingMetrics(TypedDict, total=False):
    """单个 epoch 的训练指标。"""
    epoch: int
    total_epochs: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


class ExperimentRow(TypedDict, total=False):
    """实验列表中的单行记录。"""
    experiment_name: str
    checkpoint_file: str
    checkpoint_path: str
    checkpoint_epoch: Optional[int]
    best_val_acc: Optional[float]
    result_dir: str
    updated_at: float
    source: str
    has_training_history: bool
    has_loss_curve: bool
    has_accuracy_curve: bool
    has_metrics: bool
    has_confusion_matrix: bool
    has_classification_report: bool


class ExperimentArtifacts(TypedDict, total=False):
    """实验产物集合。"""
    training_history: Dict[str, Any]
    metrics: Dict[str, Any]
    classification_report: str
    images: Dict[str, str]


class PredictionResult(TypedDict, total=False):
    """模型预测结果。"""
    prediction: int
    probabilities: List[float]
    preprocessed_image: Any
    config: Optional[Dict[str, Any]]
    checkpoint: Optional[Dict[str, Any]]


class ImagePreprocessResult(TypedDict, total=False):
    """图像预处理中间结果（用于调试）。"""
    gray: Any
    gray_corrected: Any
    fg: Any
    raw_mask: Any
    mask: Any
    crop: Any
    final_28: Any
    tensor: Any
    # 上传图片特定字段
    uploaded_gray: Optional[Any]
    uploaded_gray_corrected: Optional[Any]
    uploaded_fg: Optional[Any]
    uploaded_raw_mask: Optional[Any]
    uploaded_mask: Optional[Any]
    uploaded_crop: Optional[Any]
    uploaded_final_28: Optional[Any]
