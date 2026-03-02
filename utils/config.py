from argparse import Namespace
from enum import Enum


class EMGDataset(Enum):
    kaggle = 1
    mendeley = 2  # TODO: mendeley dataset not yet implemented fully


DEFAULT_CONFIG = {
    "project_name": "OLSN-ML",
    "run_name": "test_run",
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 1000,
    "train_samples": 128,
    "val_samples": 16,
    "dataset": EMGDataset.kaggle,
}


def update_config(args: Namespace):
    CONFIG = DEFAULT_CONFIG
    CONFIG["project_name"] = args.project_name
    CONFIG["run_name"] = args.run_name
    CONFIG["lr"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["epochs"] = args.epochs
    CONFIG["train_samples"] = args.train_samples
    CONFIG["val_samples"] = args.val_samples

    if args.dataset == "kaggle":
        CONFIG["dataset"] = EMGDataset.kaggle
    elif args.dataset == "mendeley":
        CONFIG["dataset"] = EMGDataset.mendeley

    return CONFIG
