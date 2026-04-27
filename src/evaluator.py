import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.models import MLP
from src.utils import (
    ensure_dir,
    plot_confusion_matrix,
    resolve_device,
    save_json,
    save_text,
    save_wrong_samples,
)


class BPEvaluator:
    def __init__(self, config, test_set):
        self.config = config
        self.test_set = test_set
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
        self.test_loader = DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        ensure_dir(config.experiment_result_dir)

    def _load_checkpoint(self):
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(
                f"未找到模型文件：{self.config.checkpoint_path}，请先执行训练。"
            )

        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"已加载最佳模型：{self.config.checkpoint_path}")

    def run(self):
        self._load_checkpoint()
        self.model.eval()

        all_preds = []
        all_labels = []
        wrong_samples = []
        class_ids = list(range(self.config.num_classes))
        class_names = list(self.config.resolved_class_names)

        sample_offset = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                mismatches = preds.ne(labels).cpu().tolist()
                for i, is_wrong in enumerate(mismatches):
                    if is_wrong:
                        sample_index = sample_offset + i
                        wrong_samples.append(
                            {
                                "image": self.test_set.raw_images[sample_index].cpu(),
                                "true": int(labels[i].cpu().item()),
                                "pred": int(preds[i].cpu().item()),
                            }
                        )
                sample_offset += labels.size(0)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=class_ids)

        report_text = classification_report(
            all_labels,
            all_preds,
            labels=class_ids,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        report_dict = classification_report(
            all_labels,
            all_preds,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "accuracy": round(float(accuracy), 6),
            "precision_macro": round(float(precision), 6),
            "recall_macro": round(float(recall), 6),
            "f1_macro": round(float(f1), 6),
            "class_names": class_names,
            "confusion_matrix": np.asarray(cm).tolist(),
            "classification_report": report_dict,
        }

        result_dir = self.config.experiment_result_dir
        save_json(metrics, os.path.join(result_dir, "metrics.json"))
        save_text(report_text, os.path.join(result_dir, "classification_report.txt"))

        plot_confusion_matrix(
            confusion_matrix=np.asarray(cm),
            class_names=class_names,
            save_path=os.path.join(result_dir, "confusion_matrix.png"),
        )

        save_wrong_samples(
            wrong_samples=wrong_samples,
            save_path=os.path.join(result_dir, "wrong_samples.png"),
            max_items=self.config.max_wrong_samples,
        )

        print(f"测试集 Accuracy : {accuracy:.4f}")
        print(f"测试集 Precision: {precision:.4f}")
        print(f"测试集 Recall   : {recall:.4f}")
        print(f"测试集 F1-score : {f1:.4f}")
        print(f"评估结果已保存到：{result_dir}")
