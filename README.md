# BP-DigitLab

基于 PyTorch 实现的 BP/MLP 手写数字识别课程实验项目，主模型保持为多层感知机，不改成 CNN。项目在原有 `配置文件 + Dataset + Feature + Model + Trainer + Evaluator` 架构上做了低侵入优化，重点增强了数据预处理、特征标准化、MLP 结构可配置性和训练策略，以便稳定提升 MNIST 准确率并支持实验对比。

## 项目结构

```text
BP-DigitLab/
├── main.py
├── hparams/
│   ├── bp_mnist_baseline.yaml
│   ├── bp_mnist_tuned_v1.yaml
│   ├── bp_mnist_tuned_v2.yaml
│   ├── bp_mnist_pixel.yaml
│   ├── bp_mnist_projection.yaml
│   └── bp_mnist_improved.yaml
├── src/
│   ├── hparams.py
│   ├── datasets.py
│   ├── features.py
│   ├── models.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── checkpoints/
├── results/
├── README.md
└── requirements.txt
```

## 主要增强点

- 数据集构建支持：
  - 基于训练集统计量的图像标准化；
  - 基于训练集统计量的特征标准化；
  - 仅训练阶段启用的数据增强；
  - train/val/test 共用训练集统计量。
- 特征工程支持：
  - `pixel`
  - `pixel_projection`
  - `pixel_projection_profile`（轻量扩展特征）
- MLP 模型支持：
  - 多隐层结构，如 `[256, 128]`
  - `relu / sigmoid / tanh`
  - `dropout`
  - `batch_norm`
  - `kaiming / xavier` 初始化
- 训练策略支持：
  - `sgd / sgd_momentum / adam / adamw`
  - `StepLR / ReduceLROnPlateau`
  - `CrossEntropyLoss + label smoothing`
  - `early stopping`
  - `gradient clipping`

## 安装环境

建议 Python 3.9 及以上：

```bash
pip install -r requirements.txt
```

## 训练命令

进入项目目录：

```bash
cd D:\project\codex_project\BP-DigitLab
```

训练 baseline：

```bash
python main.py --mode train --config hparams/bp_mnist_baseline.yaml
```

训练 tuned_v1：

```bash
python main.py --mode train --config hparams/bp_mnist_tuned_v1.yaml
```

训练 tuned_v2：

```bash
python main.py --mode train --config hparams/bp_mnist_tuned_v2.yaml
```

也可以继续使用原有实验配置：

```bash
python main.py --mode train --config hparams/bp_mnist_pixel.yaml
python main.py --mode train --config hparams/bp_mnist_projection.yaml
python main.py --mode train --config hparams/bp_mnist_improved.yaml
```

## 测试命令

```bash
python main.py --mode test --config hparams/bp_mnist_tuned_v2.yaml
```

如果要测试其他实验组，把配置文件改成对应 YAML 即可。

## 配置文件说明

- `bp_mnist_baseline.yaml`
  - 尽量贴近原始实现，适合作为课程实验对照组。
- `bp_mnist_tuned_v1.yaml`
  - 增加训练集统计标准化、投影特征、双隐层和 Adam。
- `bp_mnist_tuned_v2.yaml`
  - 在 tuned_v1 基础上增加 batch norm、dropout、数据增强、AdamW、ReduceLROnPlateau、label smoothing。
- `bp_mnist_pixel.yaml`
  - 原始像素 + SGD 的基础组。
- `bp_mnist_projection.yaml`
  - 原始像素 + 行列投影的特征组。
- `bp_mnist_improved.yaml`
  - 与课程设计 D 组目标相对应的增强配置。

## 结果输出位置

每次实验会在 `results/<experiment_name>/` 下保存：

- `training_history.json`
- `data_stats.json`
- `loss_curve.png`
- `accuracy_curve.png`
- `train_curve.png`
- `metrics.json`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `wrong_samples.png`

最佳模型保存在：

```text
checkpoints/<experiment_name>_best.pth
```

## 推荐实验顺序

1. 先跑 `bp_mnist_baseline.yaml`，建立原始对照结果。
2. 再跑 `bp_mnist_tuned_v1.yaml`，观察标准化、投影特征和更合理网络结构带来的提升。
3. 最后跑 `bp_mnist_tuned_v2.yaml`，观察 BN、dropout、增强、调度器和 label smoothing 的综合效果。
4. 如果报告需要与课程原始分组对齐，再补跑 `bp_mnist_pixel.yaml`、`bp_mnist_projection.yaml`、`bp_mnist_improved.yaml`。

## 哪些改动主要为了提准确率

- 训练集统计标准化
- 投影特征
- 更合理的隐层结构
- batch norm
- dropout
- Adam / AdamW
- 学习率调度器
- label smoothing
- 训练集增强

## 哪些改动主要为了更好做实验对比

- 配置项统一管理
- 训练历史与数据统计保存
- 统一的结果输出目录
- 更完整的指标、混淆矩阵和错误样本保存
- 同时保留旧配置与新配置，方便横向比较
