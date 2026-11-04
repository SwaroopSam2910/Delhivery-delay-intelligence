import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

def clean_shipments(
    raw: pd.DataFrame,
    medians: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:

    clean_df = raw.drop_duplicates().copy()

    clean_df["service_type"] = (
        clean_df["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    ).astype("float64")

    clean_df.drop(columns=["declared_value"], inplace=True)

    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"]
    )

    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = np.nan

    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            column: float(clean_df[column].median())
            for column in median_columns
        }
    else:
        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    for column in median_columns:
        clean_df[column] = clean_df[column].fillna(medians[column])

    return clean_df, medians

def build_features(
    clean: pd.DataFrame,
    hubs: pd.DataFrame
) -> pd.DataFrame:

    features_df = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one"
    )

    features_df["promised_hours"] = (
        features_df["service_type"]
        .map(PROMISED_HOURS)
        .astype(int)
    )

    features_df["is_cod"] = (
        features_df["payment_mode"] == "COD"
    ).astype(int)

    features_df["is_weekend"] = (
        features_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    features_df["dispatch_hour"] = (
        features_df["dispatch_time"].dt.hour
    ).astype(int)

    features_df["log_declared_value"] = np.log(
        features_df["declared_value_inr"]
    )

    return features_df