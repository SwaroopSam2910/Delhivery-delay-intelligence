
import os
import numpy as np
import pandas as pd

from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import FEATURES


def calculate_separation_ratio(scaled_data, cluster_labels):
    """
    Calculate mean across-cluster distance divided by
    mean within-cluster distance.

    All unordered pairs of hubs are considered.
    """

    first_indices, second_indices = np.triu_indices(
        len(scaled_data),
        k=1
    )

    distances = np.linalg.norm(
        scaled_data[first_indices] - scaled_data[second_indices],
        axis=1
    )

    same_cluster = (
        cluster_labels[first_indices]
        == cluster_labels[second_indices]
    )

    within_distances = distances[same_cluster]
    across_distances = distances[~same_cluster]

    across_mean = across_distances.mean()
    within_mean = within_distances.mean()

    return float(across_mean / within_mean)


def main():

    os.makedirs("outputs", exist_ok=True)

    data = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    hub_profile = (
        data.groupby(
            "hub_id",
            sort=True
        )
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean")
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    hub_profile_output = hub_profile.copy()

    profile_numeric_columns = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]

    hub_profile_output[
        profile_numeric_columns
    ] = hub_profile_output[
        profile_numeric_columns
    ].round(4)

    hub_profile_output.to_csv(
        "outputs/hub_profile.csv",
        index=False
    )

    print("\n========== HUB PROFILE ==========")
    print(hub_profile_output.to_string(index=False))

    profile_features = [
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops"
    ]

    profile_matrix = hub_profile[
        profile_features
    ].values

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        profile_matrix
    )

    separation_rows = []

    clustering_results = {}

    for k in [2, 3, 4, 5]:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(scaled_data)

        clustering_results[k] = labels

        separation_ratio = calculate_separation_ratio(
            scaled_data,
            labels
        )

        cluster_sizes = sorted(
            np.bincount(labels).tolist(),
            reverse=True
        )

        separation_rows.append({
            "K": k,
            "across_over_within": round(
                separation_ratio, 4
            ),
            "cluster_sizes": "-".join(
                map(str, cluster_sizes)
            )
        })

    separation_df = pd.DataFrame(
        separation_rows
    )

    separation_df.to_csv(
        "outputs/part_d_separation.csv",
        index=False
    )

    print("\n========== CLUSTER SEPARATION ==========")
    print(separation_df.to_string(index=False))

    final_labels = clustering_results[3]

    hub_profile["cluster"] = final_labels

    cluster_delay_rates = (
        hub_profile.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
    )

    segment_names = [
        "Standard",
        "Watch",
        "Escalate"
    ]

    cluster_to_segment = {
        cluster_id: segment
        for cluster_id, segment in zip(
            cluster_delay_rates.index,
            segment_names
        )
    }

    hub_profile["segment"] = (
        hub_profile["cluster"]
        .map(cluster_to_segment)
    )

    print("\n========== CLUSTER DELAY RATES ==========")
    print(cluster_delay_rates)

    hub_segments = hub_profile[
        ["hub_id", "segment"]
    ].merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "hub_tier"
            ]
        ],
        on="hub_id",
        how="left",
        validate="one_to_one"
    )

    hub_segments = hub_segments.sort_values(
        "hub_id"
    ).reset_index(drop=True)

    hub_segments = hub_segments[
        [
            "hub_id",
            "segment",
            "hub_city",
            "hub_tier"
        ]
    ]

    hub_segments.to_csv(
        "outputs/hub_segments.csv",
        index=False
    )

    print("\n========== HUB SEGMENTS ==========")
    print(hub_segments.to_string(index=False))

    segment_profile_rows = []

    for segment in segment_names:

        segment_df = hub_profile[
            hub_profile["segment"] == segment
        ]

        segment_profile_rows.append({
            "segment": segment,
            "hubs": len(segment_df),
            "mean_delivery_hours": segment_df[
                "mean_delivery_hours"
            ].mean(),
            "delay_rate": segment_df[
                "delay_rate"
            ].mean(),
            "mean_distance_km": segment_df[
                "mean_distance_km"
            ].mean(),
            "mean_n_stops": segment_df[
                "mean_n_stops"
            ].mean()
        })

    segment_profile = pd.DataFrame(
        segment_profile_rows
    )

    segment_profile[
        profile_numeric_columns
    ] = segment_profile[
        profile_numeric_columns
    ].round(4)

    segment_profile.to_csv(
        "outputs/segment_profile.csv",
        index=False
    )

    print("\n========== SEGMENT PROFILES ==========")
    print(segment_profile.to_string(index=False))

    tier_expected_segment = {
        1: "Standard",
        2: "Watch",
        3: "Escalate"
    }

    expected_segments = hub_segments[
        "hub_tier"
    ].map(tier_expected_segment)

    mismatches = hub_segments[
        hub_segments["segment"] != expected_segments
    ].copy()

    mismatches.to_csv(
        "outputs/tier_vs_segment_mismatch.csv",
        index=False
    )

    print("\n========== TIER / SEGMENT MISMATCH ==========")
    print(mismatches.to_string(index=False))

    print("\n========== PART D SEGMENTATION COMPLETED ==========")

    print("\nGenerated files:")

    print("- outputs/hub_profile.csv")
    print("- outputs/part_d_separation.csv")
    print("- outputs/hub_segments.csv")
    print("- outputs/segment_profile.csv")
    print("- outputs/tier_vs_segment_mismatch.csv")


if __name__ == "__main__":
    main()
