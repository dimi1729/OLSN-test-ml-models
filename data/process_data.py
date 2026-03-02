import os
import random

import polars as pl

from utils.config import EMGDataset


def create_data_df(path: str, dataset: EMGDataset) -> pl.DataFrame:
    if dataset == EMGDataset.kaggle:
        assert os.path.isfile(path), f"File not found: {path}"
        return pl.read_csv(path)
    elif dataset == EMGDataset.mendeley:
        assert os.path.isdir(path), (
            f"Mendeley data should be a directory of 40 csv files, instead got {path}"
        )

        # Get all CSV files in the directory
        csv_files = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")
        ]

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in directory: {path}")

        # Read all CSV files and concatenate them
        dfs = [pl.read_csv(csv_file) for csv_file in csv_files]
        combined_df = pl.concat(dfs, how="vertical")

        return combined_df


def generate_test_train_split(
    df: pl.DataFrame, train_fraction: float, val_fraction: float, test_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split the dataframe into train, validation, and test sets based on subject labels.
    Will remove label column after splitting

    df should be the raw EMG-data csv, fractions should add to 1
    """
    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-9, (
        "Fractions must sum to 1.0"
    )

    unique_labels = df["label"].unique().to_list()
    random.shuffle(unique_labels)

    n_labels = len(unique_labels)
    n_train = int(n_labels * train_fraction)
    n_val = int(n_labels * val_fraction)

    train_labels = unique_labels[:n_train]
    val_labels = unique_labels[n_train : n_train + n_val]
    test_labels = unique_labels[n_train + n_val :]

    train_df = df.filter(pl.col("label").is_in(train_labels))
    val_df = df.filter(pl.col("label").is_in(val_labels))
    test_df = df.filter(pl.col("label").is_in(test_labels))

    train_df = train_df.drop("label")
    val_df = val_df.drop("label")
    test_df = test_df.drop("label")

    return train_df, val_df, test_df


def generate_time_series_for_one_result(
    data: pl.DataFrame, time_interval: int
) -> tuple[pl.DataFrame, int]:
    """
    Basically you input the df for the training data, and time window (in ms).
    Since the rows are not exactly spaced by 1ms (there are some gaps) but they are
    approximate, just assume each row is 1ms.
    Each class has about 3,000 ms of continuous data (according to txt description).
    If you do longer than what exists, then you will get an error.

    The output is a consecutive amount of data, all corresponding to the same output.
    Since the class is constant, it is returned seperately as an int
    """
    unique_classes = data["class"].unique().to_list()
    selected_class = random.choice(unique_classes)

    class_data = data.filter(pl.col("class") == selected_class)

    num_rows_available = len(class_data)

    if num_rows_available < time_interval:
        raise ValueError(
            f"Requested {time_interval} rows but only {num_rows_available} rows "
            f"available for class {selected_class}"
        )

    # Pick a random starting index
    max_start_index = num_rows_available - time_interval
    start_index = random.randint(0, max_start_index)

    time_series = class_data.slice(start_index, time_interval)

    time_series = time_series.drop("class")
    time_series = time_series.drop("time")

    assert time_series.shape == (time_interval, 8), f"Shape is {time_series.shape}"

    return (time_series, int(selected_class))


if __name__ == "__main__":
    print("=" * 60)
    print("Testing create_data_df function")
    print("=" * 60)

    # Test with Kaggle dataset
    emg_data_path = "/home/dimi1729/Documents/OLSN-test-ml-models/data/EMG-data.csv"
    df = create_data_df(emg_data_path, EMGDataset.kaggle)
    print(f"\nKaggle dataset loaded: {len(df)} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")

    # Test train/val/test split
    print("\n" + "=" * 60)
    print("Testing train/val/test split")
    print("=" * 60)
    train_df, val_df, test_df = generate_test_train_split(df, 0.7, 0.15, 0.15)

    print(f"\nTrain set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    # Test time series generation
    print("\n" + "=" * 60)
    print("Testing time series generation")
    print("=" * 60)
    time_series, class_label = generate_time_series_for_one_result(train_df, 1000)
    print(f"\nGenerated time series: shape {time_series.shape}, class {class_label}")
    print(time_series.head(5))
