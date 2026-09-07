import os
import pandas as pd
import numpy as np
from pathlib import Path


BODY_PARTS = [
    "chest",
    "forearm",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist"
]

CONTEXT_BODY_PARTS = [
    "chest",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist"
]

ACTIVITIES = [
    "lying",
    "running",
    "sitting",
    "standing",
    "walking"
]

WINDOW_SIZE = 150
STRIDE = 75
N_DATASETS = 15


OUTPUT_DIR = Path(
    "./dataset_with_context/dataset_with_light/features_with_light/"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# DATAFRAME
# ============================================================

def build_dataframe(
    file_path,
    body_part,
    sensor,
    keep_timestamp=False
):
    df = pd.read_csv(file_path)

    new_columns = {}

    for column in df.columns:

        if column == "id":
            continue

        if "attr_time" in column:

            if keep_timestamp:
                new_columns[column] = "timestamp"

            continue

        new_columns[column] = (
            f"{body_part}_{sensor}_{column.replace('attr_', '')}"
        )

    if not keep_timestamp:

        timestamp_columns = [
            column
            for column in df.columns
            if "attr_time" in column
        ]

        df.drop(
            columns=timestamp_columns,
            inplace=True
        )

    df.rename(
        columns=new_columns,
        inplace=True
    )

    return df


# ============================================================
# SENSOR DATAFRAME
# ============================================================

def build_sensor_dataframe(
    sensor_path,
    sensor
):
    sensor_df = None

    files = sorted(os.listdir(sensor_path))

    for body_part, filename in zip(
        BODY_PARTS if sensor == "acc" or sensor == "gyr" or sensor == "mag"
        else CONTEXT_BODY_PARTS,
        files
    ):

        file_path = os.path.join(
            sensor_path,
            filename
        )

        df = build_dataframe(
            file_path,
            body_part,
            sensor
        )

        if sensor_df is None:

            sensor_df = df

        else:

            sensor_df = sensor_df.merge(
                df,
                on="id",
                how="inner"
            )

    return sensor_df


# ============================================================
# ACTIVITY DATAFRAME
# ============================================================

def build_activity_dataframe(
    activity_path,
    activity_name
):

    sensor_dataframes = {}

    sensor_folders = sorted(
        os.listdir(activity_path)
    )

    for sensor_folder in sensor_folders:

        sensor_path = os.path.join(
            activity_path,
            sensor_folder
        )

        sensor_folder_lower = (
            sensor_folder.lower()
        )

        if "acc" in sensor_folder_lower:
            sensor = "acc"

        elif "gyr" in sensor_folder_lower:
            sensor = "gyr"

        elif "mag" in sensor_folder_lower:
            sensor = "mag"

        elif "lig" in sensor_folder_lower:
            sensor = "lig"

        else:
            continue

        sensor_dataframes[sensor] = (
            build_sensor_dataframe(
                sensor_path,
                sensor
            )
        )

    # ========================================================
    # BASELINE
    # ========================================================

    activity_df = sensor_dataframes["acc"]

    activity_df = activity_df.merge(
        sensor_dataframes["gyr"],
        on="id",
        how="inner"
    )

    activity_df = activity_df.merge(
        sensor_dataframes["mag"],
        on="id",
        how="inner"
    )

    # ========================================================
    # LIGHT
    # ========================================================

    if "lig" in sensor_dataframes:

        activity_df = activity_df.merge(
            sensor_dataframes["lig"],
            on="id",
            how="inner"
        )

    activity_df["activity"] = activity_name

    return activity_df


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(subject_folder):

    dataset = []

    activity_folders = sorted(
        os.listdir(subject_folder)
    )

    for activity_folder in activity_folders:

        activity_path = os.path.join(
            subject_folder,
            activity_folder
        )

        if not os.path.isdir(activity_path):
            continue

        print(
            f"Processing activity: "
            f"{activity_folder}"
        )

        activity_df = build_activity_dataframe(
            activity_path,
            activity_folder
        )

        dataset.append(activity_df)

    dataset = pd.concat(
        dataset,
        ignore_index=True
    )

    columns = dataset.columns.tolist()

    columns.insert(
        1,
        columns.pop(
            columns.index("activity")
        )
    )

    dataset = dataset[columns]

    return dataset


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_standard_features(
    window,
    columns
):

    features = {}

    for col in columns:

        signal = window[col]

        features[f"{col}_mean"] = signal.mean()
        features[f"{col}_std"] = signal.std()
        features[f"{col}_min"] = signal.min()
        features[f"{col}_max"] = signal.max()

        if signal.notna().any():

            features[f"{col}_rms"] = np.sqrt(
                np.nanmean(
                    signal.to_numpy() ** 2
                )
            )

        else:

            features[f"{col}_rms"] = np.nan

        features[f"{col}_kurtosis"] = (
            signal.kurtosis()
        )

        features[f"{col}_skew"] = (
            signal.skew()
        )

    return features


def extract_light_features(
    window,
    body_parts
):

    mean_values = []
    std_values = []
    min_values = []
    max_values = []

    for body_part in body_parts:

        column = f"{body_part}_lig_light"

        if column not in window.columns:
            continue

        signal = window[column].dropna()

        if len(signal) == 0:
            continue

        mean_values.append(
            signal.mean()
        )

        std_values.append(
            signal.std()
        )

        min_values.append(
            signal.min()
        )

        max_values.append(
            signal.max()
        )

    features = {}

    features["light_mean"] = (
        np.mean(mean_values)
        if mean_values
        else np.nan
    )

    features["light_std"] = (
        np.mean(std_values)
        if std_values
        else np.nan
    )

    features["light_min"] = (
        np.min(min_values)
        if min_values
        else np.nan
    )

    features["light_max"] = (
        np.max(max_values)
        if max_values
        else np.nan
    )

    return features


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_id):

    input_path = (
        f"./dataset_with_context/"
        f"dataset_with_light/"
        f"dataset{dataset_id}.csv"
    )

    print(
        f"Processing {input_path}..."
    )

    df = pd.read_csv(input_path)

    standard_columns = [
        col
        for col in df.columns
        if "_lig_" not in col
        and col not in ["id", "activity"]
    ]

    light_body_parts = [
        body_part
        for body_part in CONTEXT_BODY_PARTS
        if f"{body_part}_lig_light" in df.columns
    ]

    all_features = []

    for activity in ACTIVITIES:

        activity_data = df[
            df["activity"] == activity
        ].reset_index(drop=True)

        for start in range(
            0,
            len(activity_data) - WINDOW_SIZE + 1,
            STRIDE
        ):

            window = activity_data.iloc[
                start:start + WINDOW_SIZE
            ]

            window_features = extract_standard_features(
                window,
                standard_columns
            )

            if light_body_parts:

                window_features.update(
                    extract_light_features(
                        window,
                        light_body_parts
                    )
                )

            window_features["activity"] = activity

            window_features["window_start"] = (
                window["id"].iloc[0]
            )

            window_features["window_end"] = (
                window["id"].iloc[-1]
            )

            all_features.append(
                window_features
            )

    features_df = pd.DataFrame(
        all_features
    )

    metadata_columns = [
        "activity",
        "window_start",
        "window_end"
    ]

    feature_columns = [
        col
        for col in features_df.columns
        if col not in metadata_columns
    ]

    features_df = features_df[
        metadata_columns + feature_columns
    ]

    output_path = (
        OUTPUT_DIR /
        f"dataset{dataset_id}_features.csv"
    )

    features_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"  Original samples: {len(df)}"
    )

    print(
        f"  Generated windows: "
        f"{len(features_df)}"
    )

    print(
        f"  Number of features: "
        f"{len(feature_columns)}"
    )

    print(
        f"  Saved to: {output_path}"
    )

    return features_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for i in range(2, 16):

        print(
            f"\nProcessing subject {i}..."
        )

        subject_folder = (
            f"./realworld2016_dataset/"
            f"proband{i}/baseline"
        )

        dataset = build_dataset(
            subject_folder
        )

        output_path = (
            f"./dataset_with_context/"
            f"dataset_with_light/"
            f"dataset{i}.csv"
        )

        dataset.to_csv(
            output_path,
            index=False
        )

        feature_cols = [
            c
            for c in dataset.columns
            if c not in ["id", "activity"]
        ]

        print(
            f"Number of sensor columns: "
            f"{len(feature_cols)}"
        )

        print(
            f"Dataset saved: {output_path}"
        )

        process_dataset(i)
