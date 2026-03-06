from argparse import Namespace
from enum import Enum


class EMGDataset(Enum):
    kaggle = 1
    mendeley = 2  # TODO: mendeley dataset not yet implemented fully


DATASET_CLASS_CONFIG: dict[EMGDataset, dict] = {
    EMGDataset.kaggle: {
        "num_channels": 8,
        "all_classes": list(range(0, 8)),
        "good_classes": list(range(1, 8)),  # class 0 is for unmarked_data
        "class_to_idx": {  # Really annoying hack I have to use since cross entropy loss does not like when you classes not starting with 0
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
        },
        "idx_to_class": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
        },  # Reverse mapping for predictions
        "class_names": [
            "unmarked data",
            "hand at rest",
            "hand clenched in a fist",
            "wrist flexion",
            "wrist extension",
            "radial deviations",
            "ulnar deviations",
            "extended palm",
        ],
    },
    EMGDataset.mendeley: {
        # TODO: implement
        "num_channels": 0,
        "good_classes": [],
        "class_to_idx": {},
        "idx_to_class": {},
    },
}

DEFAULT_CONFIG = {
    "project_name": "OLSN-ML",
    "run_name": "test_run",
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 1000,
    "train_samples": 128,
    "val_samples": 16,
    "dataset": EMGDataset.kaggle,
    "dataset_config": DATASET_CLASS_CONFIG[EMGDataset.kaggle].copy(),
    "use_wandb": True,
}


def update_config(args: Namespace):
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["project_name"] = args.project_name
    CONFIG["run_name"] = args.run_name
    CONFIG["lr"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["epochs"] = args.epochs
    CONFIG["train_samples"] = args.train_samples
    CONFIG["val_samples"] = args.val_samples
    CONFIG["use_wandb"] = not args.no_wandb

    if args.dataset == "kaggle":
        CONFIG["dataset"] = EMGDataset.kaggle
        CONFIG["dataset_config"] = DATASET_CLASS_CONFIG[EMGDataset.kaggle].copy()
    elif args.dataset == "mendeley":
        CONFIG["dataset"] = EMGDataset.mendeley
        CONFIG["dataset_config"] = DATASET_CLASS_CONFIG[EMGDataset.mendeley].copy()

    return CONFIG
