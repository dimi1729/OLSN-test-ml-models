import os
import random

import polars as pl


def generate_test_train_split(train_fraction: float): ...


def generate_time_series_for_one_result(
    data: pl.DataFrame, time_interval: int
) -> tuple[pl.DataFrame, int]:
    """
    Basically you input the df for the training data, and time window (in ms)
    Each class has about 3,000 ms of continuous data (according to txt description).
    If you do longer than what exists, then you will get an error.

    The output is a consecutive amount of data, all corresponding to the same output.
    Since the class is constant, it is returned seperately as an int
    """
    unique_classes = data["class"].unique().to_list()
    selected_class = random.choice(unique_classes)

    class_data = data.filter(pl.col("class") == selected_class)

    min_time: int = class_data["time"].min()
    max_time: int = class_data["time"].max()

    if (max_time - min_time) < time_interval:
        raise ValueError(
            f"Time interval {time_interval}ms is longer than available data "
            f"({max_time - min_time}ms) for class {selected_class}"
        )

    max_start_time = max_time - time_interval
    start_time = random.uniform(min_time, max_start_time)
    end_time = start_time + time_interval

    time_series = class_data.filter(
        (pl.col("time") >= start_time) & (pl.col("time") <= end_time)
    )

    time_series = time_series.drop("class")

    return (time_series, int(selected_class))


if __name__ == "__main__":
    emg_data_path = "/home/dimi1729/Documents/OLSN-test-ml-models/data/EMG-data.csv"

    df = pl.read_csv(emg_data_path)
    df = df.drop("label")  # test train will drop label, for now im lazy
    print(generate_time_series_for_one_result(df, 1000))
