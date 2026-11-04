
import os
import numpy as np
import pandas as pd

from src.shipment_prep import (
    PROMISED_HOURS,
    FEATURES,
    clean_shipments,
    build_features
)


def main():

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw_df = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw_df)

    deduplicated_raw = raw_df.drop_duplicates().copy()

    rows_after_dedup = len(deduplicated_raw)
    duplicate_rows_removed = raw_rows - rows_after_dedup

    missing_distance_before = (
        deduplicated_raw["distance_km"].isna().sum()
    )

    missing_weight_before = (
        deduplicated_raw["parcel_weight_kg"].isna().sum()
    )

    missing_declared_value_before = (
        deduplicated_raw["declared_value"].isna().sum()
    )

    missing_payment_mode_before = (
        deduplicated_raw["payment_mode"].isna().sum()
    )

    sentinel_weight_rows = (
        deduplicated_raw["parcel_weight_kg"] > 100
    ).sum()

    service_spellings_before = (
        deduplicated_raw["service_type"].nunique()
    )

    clean_df, medians = clean_shipments(raw_df)

    service_spellings_after = (
        clean_df["service_type"].nunique()
    )

    modelling_table = build_features(clean_df, hubs)

    modelling_table["is_delayed"] = (
        modelling_table["delivery_hours"]
        > modelling_table["promised_hours"]
    ).astype(int)

    rows_after_hub_merge = len(modelling_table)

    modelling_table.to_csv(
        "data/modelling_table.csv",
        index=False
    )

    declared_value_mean = (
        modelling_table["declared_value_inr"].mean()
    )

    declared_value_median = (
        modelling_table["declared_value_inr"].median()
    )

    log_declared_value = np.log(
        modelling_table["declared_value_inr"]
    )

    log_declared_value_mean = log_declared_value.mean()
    log_declared_value_median = log_declared_value.median()

    delivery_hours = modelling_table["delivery_hours"]

    hours_q1 = delivery_hours.quantile(0.25)
    hours_q3 = delivery_hours.quantile(0.75)

    iqr = hours_q3 - hours_q1

    lower_bound = hours_q1 - (1.5 * iqr)
    upper_bound = hours_q3 + (1.5 * iqr)

    hours_outlier_count = (
        (delivery_hours < lower_bound)
        | (delivery_hours > upper_bound)
    ).sum()

    delayed_count = modelling_table["is_delayed"].sum()

    delay_rate = modelling_table["is_delayed"].mean()

    nulls_in_features_and_target = (
        modelling_table[
            FEATURES + ["delivery_hours", "is_delayed"]
        ].isna().sum().sum()
    )

    checks = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": rows_after_dedup,

        "median_distance_km": round(medians["distance_km"], 4),
        "median_parcel_weight_kg": round(
            medians["parcel_weight_kg"], 4
        ),
        "median_declared_value_inr": round(
            medians["declared_value_inr"], 4
        ),

        "missing_distance_km_before": missing_distance_before,
        "missing_parcel_weight_before": missing_weight_before,
        "missing_declared_value_before": missing_declared_value_before,
        "missing_payment_mode_before": missing_payment_mode_before,

        "sentinel_weight_rows": sentinel_weight_rows,

        "service_type_spellings_before": service_spellings_before,
        "service_type_spellings_after": service_spellings_after,

        "rows_after_hub_merge": rows_after_hub_merge,
        "nulls_in_features_and_target": nulls_in_features_and_target,

        "delayed_count": delayed_count,
        "delay_rate": round(delay_rate, 4),

        "hours_q1": round(hours_q1, 4),
        "hours_q3": round(hours_q3, 4),
        "hours_lower_bound": round(lower_bound, 4),
        "hours_upper_bound": round(upper_bound, 4),
        "hours_outlier_count": hours_outlier_count,

        "declared_value_mean_inr": round(declared_value_mean, 4),
        "declared_value_median_inr": round(declared_value_median, 4),
        "log_declared_value_mean": round(
            log_declared_value_mean, 4
        ),
        "log_declared_value_median": round(
            log_declared_value_median, 4
        )
    }

    checks_df = pd.DataFrame(
        list(checks.items()),
        columns=["metric", "value"]
    )

    checks_df.to_csv(
        "outputs/part_a_checks.csv",
        index=False
    )

    modelling_table["distance_band"] = pd.cut(
        modelling_table["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True
    )

    delay_records = []

    dimensions = [
        "service_type",
        "hub_tier",
        "distance_band"
    ]

    for dimension in dimensions:

        grouped = modelling_table.groupby(
            dimension,
            sort=True,
            observed=True
        )

        for group, group_df in grouped:

            shipments = len(group_df)
            delayed = group_df["is_delayed"].sum()

            delay_records.append({
                "dimension": dimension,
                "group": group,
                "shipments": shipments,
                "delayed": delayed,
                "delay_rate": round(
                    delayed / shipments, 4
                ),
                "mean_hours": round(
                    group_df["delivery_hours"].mean(), 4
                )
            })

    delay_by_group = pd.DataFrame(delay_records)

    delay_by_group.to_csv(
        "outputs/part_a_delay_by_group.csv",
        index=False
    )

    correlations = modelling_table.corr(
        numeric_only=True
    )

    correlations = correlations.round(4)

    correlations.to_csv(
        "outputs/part_a_correlations.csv",
        index=True,
        index_label="column"
    )

    print("\n========== PART A COMPLETED ==========")

    print("\nModelling table:", modelling_table.shape)

    print("\nCalculated medians:")
    print(medians)

    print("\nChecks:")
    print(checks_df.to_string(index=False))

    print("\nDelay by group:")
    print(delay_by_group.to_string(index=False))

    print("\nGenerated files:")
    print("- data/modelling_table.csv")
    print("- outputs/part_a_checks.csv")
    print("- outputs/part_a_delay_by_group.csv")
    print("- outputs/part_a_correlations.csv")


if __name__ == "__main__":
    main()
