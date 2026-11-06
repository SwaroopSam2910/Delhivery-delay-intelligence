
import os
import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features
)


def main():

    os.makedirs("outputs", exist_ok=True)

    new_shipments = pd.read_csv(
        "data/new_shipments.csv"
    )

    hubs = pd.read_csv(
        "data/hubs.csv"
    )

    hub_segments = pd.read_csv(
        "outputs/hub_segments.csv"
    )

    model = joblib.load(
        "models/delay_model.pkl"
    )

    threshold = float(
        pd.read_csv(
            "models/delay_threshold.csv"
        )["threshold"].iloc[0]
    )

    feature_order = pd.read_csv(
        "models/feature_order.csv"
    )["feature"].tolist()

    impute_df = pd.read_csv(
        "models/impute_values.csv"
    )

    medians = {
        row["column"]: float(row["median"])
        for _, row in impute_df.iterrows()
    }

    print("\n========== LOADED MODEL CONFIGURATION ==========")
    print("Alert threshold:", threshold)
    print("Number of features:", len(feature_order))
    print("Historical imputation medians:", medians)

    clean_df, _ = clean_shipments(
        new_shipments,
        medians=medians
    )

    features_df = build_features(
        clean_df,
        hubs
    )

    X_new = features_df[feature_order].copy()

    # Validate the model input
    if X_new.isnull().any().any():
        raise ValueError(
            "Missing values detected in model features."
        )

    if len(feature_order) != 17:
        raise ValueError(
            "Expected exactly 17 model features."
        )

    probabilities = model.predict_proba(
        X_new
    )[:, 1]

    delay_probabilities = np.round(
        probabilities,
        4
    )

    flagged = (
        delay_probabilities >= threshold
    ).astype(int)

    segment_lookup = (
        hub_segments
        .set_index("hub_id")["segment"]
    )

    segments = features_df["hub_id"].map(
        segment_lookup
    )

    if segments.isna().any():
        raise ValueError(
            "Some shipments have no matching hub segment."
        )

    conditions = [
        (flagged == 1) & (segments == "Escalate"),
        flagged == 1
    ]

    choices = [
        "Escalate to hub manager",
        "Proactive customer call"
    ]

    actions = np.select(
        conditions,
        choices,
        default="No action"
    )

    actions_df = pd.DataFrame({
        "shipment_id": features_df["shipment_id"],
        "hub_id": features_df["hub_id"],
        "segment": segments,
        "delay_probability": delay_probabilities,
        "flagged": flagged,
        "action": actions
    })

    if len(actions_df) != len(new_shipments):
        raise ValueError(
            "Shipment row count changed during prediction."
        )

    if actions_df["shipment_id"].duplicated().any():
        raise ValueError(
            "Duplicate shipment IDs detected."
        )

    if actions_df.isnull().any().any():
        raise ValueError(
            "Missing values detected in final action list."
        )

    output_path = "outputs/new_shipment_actions.csv"

    actions_df.to_csv(
        output_path,
        index=False
    )

    print("\n========== NEW SHIPMENT PREDICTIONS ==========")

    print(actions_df.to_string(index=False))

    print("\n========== OPERATIONAL SUMMARY ==========")

    print("Total shipments:", len(actions_df))

    print("Flagged shipments:", int(flagged.sum()))

    print(
        "Hub-manager escalations:",
        int((actions_df["action"] == "Escalate to hub manager").sum())
    )

    print(
        "Proactive customer calls:",
        int((actions_df["action"] == "Proactive customer call").sum())
    )

    print(
        "No-action shipments:",
        int((actions_df["action"] == "No action").sum())
    )

    print("\nAction distribution:")
    print(actions_df["action"].value_counts())

    print("\nFinal output saved:", output_path)


if __name__ == "__main__":
    main()
