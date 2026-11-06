
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import (
    FEATURES,
    clean_shipments
)


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data = pd.read_csv("data/modelling_table.csv")

    config = pd.read_csv(
        "outputs/part_c_final_config.csv"
    ).iloc[0]

    X = data[FEATURES].copy()
    y = data["is_delayed"].copy()

    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # Convert unlimited depth back to Python None
    if str(config["max_depth"]).lower() == "unlimited":
        max_depth = None
    else:
        max_depth = int(config["max_depth"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    assert len(X_train) == 840
    assert len(X_val) == 180
    assert len(X_test) == 180

    print("\n========== REBUILT DATA SPLIT ==========")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    final_model = Pipeline([
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=selector_k
            )
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        )
    ])

    final_model.fit(X_train, y_train)

    model_path = "models/delay_model.pkl"

    joblib.dump(
        final_model,
        model_path
    )

    pd.DataFrame({
        "threshold": [threshold]
    }).to_csv(
        "models/delay_threshold.csv",
        index=False
    )

    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        "models/feature_order.csv",
        index=False
    )

    raw_shipments = pd.read_csv(
        "data/raw_shipments.csv"
    )

    _, medians = clean_shipments(raw_shipments)

    impute_values = pd.DataFrame({
        "column": list(medians.keys()),
        "median": [
            round(value, 4)
            for value in medians.values()
        ]
    })

    impute_values.to_csv(
        "models/impute_values.csv",
        index=False
    )

    loaded_model = joblib.load(model_path)

    original_probabilities = (
        final_model.predict_proba(X_test)[:, 1]
    )

    reloaded_probabilities = (
        loaded_model.predict_proba(X_test)[:, 1]
    )

    max_abs_probability_diff = float(
        np.max(
            np.abs(
                original_probabilities
                - reloaded_probabilities
            )
        )
    )

    reload_check = pd.DataFrame([
        {
            "metric": "test_rows_scored",
            "value": len(X_test)
        },
        {
            "metric": "reload_max_abs_probability_diff",
            "value": round(max_abs_probability_diff, 4)
        },
        {
            "metric": "threshold",
            "value": round(threshold, 4)
        }
    ])

    reload_check.to_csv(
        "outputs/part_d_reload_check.csv",
        index=False
    )

    print("\n========== MODEL PACKAGING ==========")

    print("Model saved:", model_path)

    print("\nFinal configuration:")
    print("Selector k:", selector_k)
    print("Number of estimators:", n_estimators)
    print("Maximum depth:", max_depth)
    print("Minimum samples per leaf:", min_samples_leaf)
    print("Threshold:", threshold)

    print("\nImputation medians:")
    print(impute_values.to_string(index=False))

    print("\nReload verification:")
    print(reload_check.to_string(index=False))

    print("\nMaximum probability difference:",
          max_abs_probability_diff)

    print("\nGenerated files:")
    print("- models/delay_model.pkl")
    print("- models/delay_threshold.csv")
    print("- models/feature_order.csv")
    print("- models/impute_values.csv")
    print("- outputs/part_d_reload_check.csv")


if __name__ == "__main__":
    main()
