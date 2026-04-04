import os
import torch
from typing import Any

from data.process_data import (
    create_data_df,
    EMGTimeSeriesDataset
)
from utils.arg_parser import parser
from utils.config import update_config

if __name__ == "__main__":
    args = parser.parse_args()
    CONFIG: dict[str, Any] = update_config("inference", args)

    df = create_data_df(CONFIG["dataset_config"]["path"], CONFIG["dataset"])

    model = torch.load(CONFIG["checkpoint_to_load"])
    model.eval()

    dataset = EMGTimeSeriesDataset(
        df=df,
        time_interval=CONFIG["time_interval"],
        good_classes=CONFIG["dataset_config"]["good_classes"],
        num_channels=CONFIG["dataset_config"]["good_classes"],
        class_to_idx=CONFIG["dataset_config"]["class_to_idx"],
        num_samples=CONFIG["inference_samples"],
    )
