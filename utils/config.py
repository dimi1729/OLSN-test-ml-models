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
        "path": "data/datasets/EMG-data.csv",
    },
    EMGDataset.mendeley: {
        "num_channels": 4,
        "good_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 10 hand position classes
        "class_to_idx": {i: i for i in range(0, 10)},
        "idx_to_class": {i: i for i in range(0, 10)},
        "class_names": [
            "Palm down",
            "Extension",
            "Flexion",
            "Ulnar deviation",
            "Radial deviation",
            "Grip",
            "Abduction of all fingers",
            "Adduction of all fingers",
            "Supination",
            "Pronation",
        ],
        "path": "data/datasets/mendeley",
    },
}

DEFAULT_CONFIG = {
    "project_name": "OLSN-ML",
    "run_name": "test_run",
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 1000,
    "time_interval": 1024,  # in ms
    "train_samples": 128,
    "val_samples": 16,
    "dataset": EMGDataset.kaggle,
    "dataset_config": DATASET_CLASS_CONFIG[EMGDataset.kaggle].copy(),
    "use_wandb": True,
    "split": [0.7, 0.15, 0.15],  # train, val, test proportions
}


def update_config(args: Namespace):
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG["project_name"] = args.project_name
    CONFIG["run_name"] = args.run_name
    CONFIG["lr"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["epochs"] = args.epochs
    CONFIG["time_interval"] = args.time_interval
    CONFIG["train_samples"] = args.train_samples
    CONFIG["val_samples"] = args.val_samples
    CONFIG["use_wandb"] = not args.no_wandb

    # Validate and set split proportions
    split = args.split
    assert len(split) == 3, "Split must have exactly 3 values: train, val, test"
    assert abs(sum(split) - 1.0) < 1e-6, (
        f"Split proportions must sum to 1.0, got {sum(split)}"
    )
    assert all(0 <= s <= 1 for s in split), "Split proportions must be between 0 and 1"
    CONFIG["split"] = split

    if args.dataset == "kaggle":
        CONFIG["dataset"] = EMGDataset.kaggle
        CONFIG["dataset_config"] = DATASET_CLASS_CONFIG[EMGDataset.kaggle].copy()
    elif args.dataset == "mendeley":
        CONFIG["dataset"] = EMGDataset.mendeley
        CONFIG["dataset_config"] = DATASET_CLASS_CONFIG[EMGDataset.mendeley].copy()

    return CONFIG
