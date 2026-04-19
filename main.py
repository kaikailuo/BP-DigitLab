import argparse

from src.datasets import build_test_dataset, build_train_val_datasets
from src.evaluator import BPEvaluator
from src.hparams import BPTrainingHparams
from src.trainer import BPTrainer
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="基于 BP/MLP 的 MNIST 手写数字识别系统")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="运行模式：train 或 test",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 超参数配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = BPTrainingHparams.from_hparams(args.config)
    set_seed(config.seed)

    if args.mode == "train":
        train_set, val_set = build_train_val_datasets(config)
        trainer = BPTrainer(config=config, train_set=train_set, val_set=val_set)
        trainer.run()
    else:
        test_set = build_test_dataset(config)
        evaluator = BPEvaluator(config=config, test_set=test_set)
        evaluator.run()


if __name__ == "__main__":
    main()
