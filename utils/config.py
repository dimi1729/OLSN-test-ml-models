from enum import Enum


class EMGDataset(Enum):
    kaggle = 1
    mendeley = 2


CONFIG = {
    "lr": 0.001,
    "batch_size": 2,
    "epochs": 1000,
    "train_samples": 128,
    "val_samples": 16,
    "dataset": EMGDataset.kaggle,
}

PROJECT_NAME = "OLSN-ML"
