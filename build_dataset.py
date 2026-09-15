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
GPS_TOLERANCE = 30000

MIC_FREQUENCY = 1.5
WINDOW_DURATION_SECONDS = WINDOW_SIZE / 50
EXPECTED_MIC_SAMPLES = (
    MIC_FREQUENCY * WINDOW_DURATION_SECONDS
)


OUTPUT_DIR = Path(
    "./dataset_with_context/dataset_all/features_with_all_sensors/"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



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
    
def build_gps_dataframes(gps_path):

    gps_dataframes = {}

    files = sorted(
        os.listdir(gps_path)
    )

    for body_part, filename in zip(
        CONTEXT_BODY_PARTS,
        files
    ):

        file_path = os.path.join(
            gps_path,
            filename
        )

        df = build_dataframe(
            file_path,
            body_part,
            "gps",
            keep_timestamp=True
        )

        df = df.rename(
            columns={
                "timestamp": "timestamp",
                f"{body_part}_gps_lat": "lat",
                f"{body_part}_gps_lng": "lng"
            }
        )

        df = calculate_gps_features(df)

        df.rename(
            columns={
                "lat":
                    f"{body_part}_gps_lat",

                "lng":
                    f"{body_part}_gps_lng",

                "speed_mps":
                    f"{body_part}_gps_speed_mps",

                "gps_available":
                    f"{body_part}_gps_available"
            },
            inplace=True
        )

        gps_dataframes[body_part] = df

    return gps_dataframes

def build_sensor_dataframe(
    sensor_path,
    sensor
):
    sensor_df = None

    files = sorted(
        os.listdir(sensor_path)
    )

    body_parts = (
        BODY_PARTS
        if sensor in ["acc", "gyr", "mag"]
        else CONTEXT_BODY_PARTS
    )

    for index, (body_part, filename) in enumerate(
        zip(body_parts, files)
    ):

        file_path = os.path.join(
            sensor_path,
            filename
        )

        df = build_dataframe(
            file_path,
            body_part,
            sensor,
            keep_timestamp=(
                sensor == "acc"
                and index == 0
            )
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
    


def build_mic_dataframe(
    sensor_path
):

    mic_data = []

    files = sorted(
        os.listdir(sensor_path)
    )

    for body_part, filename in zip(
        CONTEXT_BODY_PARTS,
        files
    ):

        file_path = os.path.join(
            sensor_path,
            filename
        )

        df = pd.read_csv(
            file_path
        )

        df = df[
            [
                "attr_time",
                "attr_db"
            ]
        ].copy()

        df["attr_time"] = pd.to_numeric(
            df["attr_time"],
            errors="coerce"
        )

        df["attr_db"] = pd.to_numeric(
            df["attr_db"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["attr_time"]
        )

        df = df.sort_values(
            "attr_time"
        )

        df = df.rename(
            columns={
                "attr_time": "timestamp",
                "attr_db": f"{body_part}_mic_db"
            }
        )

        mic_data.append(
            df[
                [
                    "timestamp",
                    f"{body_part}_mic_db"
                ]
            ]
        )

    return mic_data


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

        elif "mic" in sensor_folder_lower:
            sensor = "mic"
            
        elif "lig" in sensor_folder_lower: 
            sensor = "lig"
            
        elif "gps" in sensor_folder_lower:
            sensor = "gps"

        else:
            continue

        if sensor == "mic":
            sensor_dataframes[sensor] = (
                build_mic_dataframe(
                    sensor_path
                )
            )
        elif sensor == "gps":
            sensor_dataframes["gps"] = (
                build_gps_dataframes(
                    sensor_path
                )
            )

        else:

            sensor_dataframes[sensor] = (
                build_sensor_dataframe(
                    sensor_path,
                    sensor
                )
            )

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

    activity_df["timestamp"] = pd.to_numeric(
        activity_df["timestamp"],
        errors="coerce"
    )
    
    if "gps" in sensor_dataframes:

        activity_df = activity_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        for body_part, gps_df in (
            sensor_dataframes["gps"].items()
        ):

            gps_df = gps_df[
                [
                    "timestamp",
                    f"{body_part}_gps_lat",
                    f"{body_part}_gps_lng",
                    f"{body_part}_gps_speed_mps",
                    f"{body_part}_gps_available"
                ]
            ].copy()

            gps_df = gps_df.sort_values(
                "timestamp"
            )

            activity_df = pd.merge_asof(
                activity_df,
                gps_df,
                on="timestamp",
                direction="backward",
                tolerance=GPS_TOLERANCE
            )

    if "lig" in sensor_dataframes: 
        activity_df = activity_df.merge(sensor_dataframes["lig"], on="id", how="inner" )

    activity_df["activity"] = activity_name

    return activity_df, sensor_dataframes.get("mic")

def build_dataset(
    subject_folder
):

    dataset = []
    mic_data = {}

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

        activity_df, activity_mic = (
            build_activity_dataframe(
                activity_path,
                activity_folder
            )
        )

        dataset.append(
            activity_df
        )

        if activity_mic is not None:

            mic_data[
                activity_folder
            ] = activity_mic

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

    return dataset, mic_data


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
    
def extract_gps_features(
    window,
    body_parts
):

    features = {}

    lat_values = []
    lng_values = []
    speed_values = []
    speed_max_values = []
    availability_values = []

    for body_part in body_parts:

        lat = (
            window[
                f"{body_part}_gps_lat"
            ]
            .dropna()
        )

        lng = (
            window[
                f"{body_part}_gps_lng"
            ]
            .dropna()
        )

        speed = (
            window[
                f"{body_part}_gps_speed_mps"
            ]
            .dropna()
        )

        availability = (
            window[
                f"{body_part}_gps_available"
            ]
            .fillna(0)
        )

        if len(lat) > 0:
            lat_values.append(lat.iloc[-1])

        if len(lng) > 0:
            lng_values.append(lng.iloc[-1])

        if len(speed) > 0:
            speed_values.extend(speed.tolist())

        if len(speed) > 0:
            speed_max_values.append(speed.max())

        availability_values.append(
            availability.mean()
        )

    features["gps_lat"] = (
        np.mean(lat_values)
        if lat_values
        else np.nan
    )

    features["gps_lng"] = (
        np.mean(lng_values)
        if lng_values
        else np.nan
    )

    features["gps_speed_mean"] = (
        np.mean(speed_values)
        if speed_values
        else np.nan
    )

    features["gps_speed_max"] = (
        np.max(speed_max_values)
        if speed_max_values
        else np.nan
    )

    features["gps_availability"] = (
        np.mean(availability_values)
        if availability_values
        else 0
    )

    return features


def calculate_gps_features(df):

    df = df.copy()

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lng"] = pd.to_numeric(
        df["lng"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "lat", "lng"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    dt = (
        df["timestamp"].diff()
        / 1000.0
    )

    lat1 = np.radians(df["lat"].shift(1))
    lat2 = np.radians(df["lat"])

    lon1 = np.radians(df["lng"].shift(1))
    lon2 = np.radians(df["lng"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distance = (
        2
        * 6371000
        * np.arcsin(
            np.sqrt(a)
        )
    )

    df["speed_mps"] = distance / dt
    df["gps_available"] = 1

    return df

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

def extract_mic_features(
    window_start,
    window_end,
    mic_data
):

    mean_values = []
    std_values = []
    min_values = []
    max_values = []
    availability_values = []

    for body_part, mic_df in zip(
        CONTEXT_BODY_PARTS,
        mic_data
    ):

        column = f"{body_part}_mic_db"

        mic_window = mic_df[
            (mic_df["timestamp"] >= window_start)
            & (mic_df["timestamp"] <= window_end)
        ][column].dropna()

        n_samples = len(mic_window)

        availability = min(
            n_samples / EXPECTED_MIC_SAMPLES,
            1.0
        )

        availability_values.append(
            availability
        )

        if n_samples == 0:
            continue

        mean_values.append(
            mic_window.mean()
        )

        std_values.append(
            mic_window.std(ddof=0)
        )

        min_values.append(
            mic_window.min()
        )

        max_values.append(
            mic_window.max()
        )

    features = {}

    features["mic_db_mean"] = (
        np.mean(mean_values)
        if mean_values
        else np.nan
    )

    features["mic_db_std"] = (
        np.mean(std_values)
        if std_values
        else np.nan
    )

    features["mic_db_min"] = (
        np.min(min_values)
        if min_values
        else np.nan
    )

    features["mic_db_max"] = (
        np.max(max_values)
        if max_values
        else np.nan
    )

    features["mic_availability"] = (
        np.mean(availability_values)
        if availability_values
        else 0.0
    )

    return features

def process_dataset(
    dataset_id,
    mic_data
):

    input_path = (
        f"./dataset_with_context/"
        f"dataset_all/"
        f"dataset{dataset_id}.csv"
    )

    print(
        f"Processing {input_path}..."
    )

    df = pd.read_csv(
        input_path
    )

    standard_columns = [
        col
        for col in df.columns
        if "_lig_" not in col
        and "_gps_" not in col
        and col not in ["id", "activity", "timestamp"]
    ]
    
    light_body_parts = [ body_part for body_part in CONTEXT_BODY_PARTS if f"{body_part}_lig_light" in df.columns ]
    
    gps_body_parts = [
        body_part
        for body_part in CONTEXT_BODY_PARTS
        if f"{body_part}_gps_lat" in df.columns
    ]

    all_features = []

    for activity in ACTIVITIES:

        activity_data = df[
            df["activity"] == activity
        ].reset_index(drop=True)

        activity_mic = mic_data.get(
            activity
        )

        for start in range(
            0,
            len(activity_data) - WINDOW_SIZE + 1,
            STRIDE
        ):

            window = activity_data.iloc[
                start:start + WINDOW_SIZE
            ]

            window_start = (
                window["timestamp"].iloc[0]
            )

            window_end = (
                window["timestamp"].iloc[-1]
            )

            window_features = (
                extract_standard_features(
                    window,
                    standard_columns
                )
            )
            if gps_body_parts:

                window_features.update(
                    extract_gps_features(
                        window,
                        gps_body_parts
                    )
                )

            if light_body_parts: 
                window_features.update(extract_light_features(window, light_body_parts))


            if activity_mic is not None:
                window_features.update(
                    extract_mic_features(
                        window_start,
                        window_end,
                        activity_mic
                    )
                )

            window_features["activity"] = (
                activity
            )

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
        f"  Original samples: "
        f"{len(df)}"
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
        f"  Saved to: "
        f"{output_path}"
    )

    return features_df


if __name__ == "__main__":

    for i in range(1, 16):
        print(
            f"\nProcessing subject {i}..."
        )

        subject_folder = (
            f"./realworld2016_dataset/"
            f"proband{i}/baseline"
        )

        dataset, mic_data = build_dataset(
            subject_folder
        )

        output_path = (
            f"./dataset_with_context/"
            f"dataset_all/"
            f"dataset{i}.csv"
        )

        Path(
            "./dataset_with_context/"
            "dataset_all/"
        ).mkdir(
            exist_ok=True,
            parents=True
        )

        dataset.to_csv(
            output_path,
            index=False
        )

        feature_cols = [
            c
            for c in dataset.columns
            if c not in [
                "id",
                "activity",
                "timestamp"
            ]
        ]

        print(
            f"Number of sensor columns: "
            f"{len(feature_cols)}"
        )

        print(
            f"Dataset saved: "
            f"{output_path}"
        )

        process_dataset(
            i,
            mic_data
        )
