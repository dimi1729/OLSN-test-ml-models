import os
import random

import polars as pl

from utils.config import EMGDataset


def classify_mendeley_label_group(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classify a single label group (subject) of Mendeley data by adding a "class" column.

    This processes one subject's data (approximately 1,280,000 rows) and adds class labels
    based on the timing of hand position cycles.

    Basically there should be 5 cycles of 10 different hand positions

    There are 4 seconds of changing between positions, and then 6 seconds of holding each position
    so 104 seconds per cycle 10 * 6 + 4 * 7 (4 seconds of changing are before and after each cycle)
    Theres also 30 seconds between each cycle (not flanking beginning and end)
    so 640 seconds total (104 * 5 + 30 * 4).

    These are measured at 2000Hz (2 samples/ms), but we downsample to 1000Hz (1 sample/ms)
    so there should be 640,000 rows per label after downsampling.

    Class labels correspond to hand positions:
        0: Palm down
        1: Extension
        2: Flexion
        3: Ulnar deviation
        4: Radial deviation
        5: Grip
        6: Abduction of all fingers
        7: Adduction of all fingers
        8: Supination
        9: Pronation

    Rest periods (during transitions) are removed.

    Args:
        df: DataFrame for a single label/subject with channel columns

    Returns:
        DataFrame with added "class" column
    """
    # Assert approximately 640,000 rows after downsampling (allow 5% tolerance)
    # Original was 1,280,000 at 2000Hz, now 640,000 at 1000Hz
    expected_rows = 640_000
    tolerance = 0.01
    min_rows = int(expected_rows * (1 - tolerance))
    max_rows = int(expected_rows * (1 + tolerance))
    actual_rows = len(df)

    assert min_rows <= actual_rows <= max_rows, (
        f"Expected approximately {expected_rows:,} rows (+/-{tolerance * 100}%), "
        f"but got {actual_rows:,} rows for label {df['label'][0]}"
    )

    # Timing constants (all in samples, where 1ms = 1 sample after downsampling)
    samples_per_ms = 1
    holding_duration = 6 * 1000 * samples_per_ms  # 6 seconds = 6,000 samples
    transition_duration = 4 * 1000 * samples_per_ms  # 4 seconds = 4,000 samples
    between_cycle_duration = 30 * 1000 * samples_per_ms  # 30 seconds = 30,000 samples

    # One cycle: initial transition + 10 positions (each with holding + transition)
    # But last position in cycle has no transition (goes to between-cycle rest)
    cycle_duration = (
        transition_duration + 10 * holding_duration + 9 * transition_duration
    )

    # Create class column initialized to -1 (rest/transition)
    classes = [-1] * len(df)

    # Process each cycle
    for cycle in range(5):
        cycle_start = cycle * (cycle_duration + between_cycle_duration)

        # Initial transition before first position
        current_pos = cycle_start + transition_duration

        # Process 10 positions in this cycle
        for position in range(10):
            # Mark holding period with class label
            for i in range(holding_duration):
                idx = current_pos + i
                if idx < len(classes):
                    classes[idx] = position

            current_pos += holding_duration

            # Add transition time (except after last position in cycle)
            if position < 9:
                current_pos += transition_duration

    # Add class column to dataframe
    df = df.with_columns(pl.Series("class", classes))

    # Remove rest periods (class == -1)
    df = df.filter(pl.col("class") != -1)

    return df


def classify_mendeley_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classify entire Mendeley dataset by processing each label group separately.

    Each label (subject) has its own timing cycle that needs to be processed independently.

    Args:
        df: DataFrame with label and channel columns for all subjects

    Returns:
        DataFrame with added "class" column, with rest periods removed
    """
    # Get unique labels and process each one separately
    unique_labels = df["label"].unique().sort().to_list()

    classified_dfs = []
    for label in unique_labels:
        label_df = df.filter(pl.col("label") == label)
        classified_label_df = classify_mendeley_label_group(label_df)
        classified_dfs.append(classified_label_df)

    # Concatenate all classified label groups
    result_df = pl.concat(classified_dfs, how="vertical")

    return result_df


def process_mendeley_file(filepath: str) -> pl.DataFrame:
    """
    Process a single Mendeley CSV file:
    1. Read the CSV
    2. Downsample from 2000Hz to 1000Hz (take every other row)
    3. Rename columns to channel1, channel2, etc.
    4. Extract label from filename (e.g., "1_filtered.csv" -> label=1)
    5. Add label column to dataframe

    Args:
        filepath: Full path to the CSV file

    Returns:
        DataFrame with renamed channels and label column, downsampled to 1000Hz
    """
    df = pl.read_csv(filepath)

    # Downsample from 2000Hz to 1000Hz by taking every other sample
    df = df.filter(pl.int_range(pl.len()).mod(2) == 0)

    # Rename columns to channel1, channel2, etc.
    num_columns = len(df.columns)
    new_column_names = [f"channel{i + 1}" for i in range(num_columns)]
    df = df.rename(dict(zip(df.columns, new_column_names)))

    # Extract label from filename (e.g., "1_filtered.csv" -> 1)
    filename = os.path.basename(filepath)
    label = int(filename.split("_")[0])

    # Add label column
    df = df.with_columns(pl.lit(label).alias("label"))

    return df


def create_data_df(path: str, dataset: EMGDataset) -> pl.DataFrame:
    if dataset == EMGDataset.kaggle:
        assert os.path.isfile(path), f"File not found: {path}"
        return pl.read_csv(path)
    elif dataset == EMGDataset.mendeley:
        assert os.path.isdir(path), (
            f"Mendeley data should be a directory of csv files, instead got {path}"
        )

        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in directory: {path}")

        # Process each CSV file with proper column naming and labels
        dfs = [
            process_mendeley_file(os.path.join(path, csv_file))
            for csv_file in csv_files
        ]
        combined_df = pl.concat(dfs, how="vertical")

        # Classify the Mendeley data by adding class column
        combined_df = classify_mendeley_df(combined_df)

        return combined_df


def generate_test_train_split(
    df: pl.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
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
    data: pl.DataFrame,
    time_interval: int,
    good_classes: list,
    num_channels: int,
    class_to_idx: dict[int, int],
) -> tuple[pl.DataFrame, int]:
    """
    Basically you input the df for the training data, and time window (in ms).
    Since the rows are not exactly spaced by 1ms (there are some gaps) but they are
    approximate, just assume each row is 1ms.
    Each class has about 3,000 ms of continuous data (according to txt description).
    If you do longer than what exists, then you will get an error.

    The output is a consecutive amount of data, all corresponding to the same output.
    Since the class is constant, it is returned separately as an int.

    Args:
        data: DataFrame containing the EMG data
        time_interval: Number of time steps to sample
        good_classes: List of valid class labels to sample from
        num_channels: Expected number of EMG channels
        class_to_idx: Mapping from original class labels to model indices [0, num_classes-1]

    Returns:
        Tuple of (time_series DataFrame, remapped class index for model)
    """
    unique_classes = data["class"].unique().to_list()
    assert len(set(unique_classes).intersection(set(good_classes))) >= 2, (
        f"Less than 2 classes between good classes {good_classes} and classes in dataset {unique_classes}"
    )
    unique_classes = list(set(unique_classes).intersection(set(good_classes)))
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

    assert time_series.shape == (time_interval, num_channels), (
        f"Shape is {time_series.shape}, expected ({time_interval}, {num_channels})"
    )

    # Remap the class label to model index (0 to num_classes-1)
    remapped_class = class_to_idx[selected_class]

    return (time_series, remapped_class)


if __name__ == "__main__":
    mendeley_path = (
        "/home/dimi1729/Documents/OLSN-test-ml-models/data/datasets/mendeley"
    )
    mendeley_df = create_data_df(mendeley_path, EMGDataset.mendeley)
    print(
        f"\nMendeley dataset loaded: {len(mendeley_df)} rows, {mendeley_df.shape[1]} columns"
    )
    print(f"Columns: {mendeley_df.columns}")
    print(f"Unique labels: {sorted(mendeley_df['label'].unique().to_list())}")
    print(f"Unique classes: {sorted(mendeley_df['class'].unique().to_list())}")
    print("Head")
    print(mendeley_df.head(10))

    print("Class distribution:")
    class_counts = mendeley_df.group_by("class").agg(pl.len()).sort("class")
    print(class_counts)
