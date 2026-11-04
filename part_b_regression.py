
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.shipment_prep import FEATURES


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred))
    }


def main():

    os.makedirs("outputs", exist_ok=True)

    data = pd.read_csv("data/modelling_table.csv")

    X = data[FEATURES].copy()
    y = data["delivery_hours"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    print("\n--- DATA SPLIT ---")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    train_scaled_max_abs_mean = float(
        np.abs(X_train_scaled.mean(axis=0)).max()
    )

    test_scaled_max_abs_mean = float(
        np.abs(X_test_scaled.mean(axis=0)).max()
    )

    checks = pd.DataFrame([
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "val_rows", "value": len(X_val)},
        {"metric": "test_rows", "value": len(X_test)},
        {
            "metric": "train_scaled_max_abs_mean",
            "value": round(train_scaled_max_abs_mean, 4)
        },
        {
            "metric": "test_scaled_max_abs_mean",
            "value": round(test_scaled_max_abs_mean, 4)
        }
    ])

    checks.to_csv(
        "outputs/part_b_checks.csv",
        index=False
    )

    mean_prediction = y_train.mean()

    baseline_predictions = np.full(
        len(y_val),
        mean_prediction
    )

    baseline_metrics = calculate_metrics(
        y_val,
        baseline_predictions
    )

    metrics_rows = [{
        "model": "mean_baseline",
        "split": "val",
        **baseline_metrics
    }]

    tree_depths = [1, 2, 3, 4, 6, 8, 12, None]

    tree_candidates = []
    tree_depth_rows = []

    for order, depth in enumerate(tree_depths):

        tree = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        tree.fit(X_train, y_train)

        train_predictions = tree.predict(X_train)
        val_predictions = tree.predict(X_val)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_predictions)
        )

        val_rmse = np.sqrt(
            mean_squared_error(y_val, val_predictions)
        )

        tree_candidates.append({
            "depth": depth,
            "order": order,
            "val_rmse": val_rmse,
            "model": tree
        })

        tree_depth_rows.append({
            "max_depth": (
                "unlimited" if depth is None else depth
            ),
            "train_rmse": round(float(train_rmse), 4),
            "val_rmse": round(float(val_rmse), 4)
        })

    best_tree = min(
        tree_candidates,
        key=lambda item: (item["val_rmse"], item["order"])
    )

    best_depth = best_tree["depth"]

    pd.DataFrame(tree_depth_rows).to_csv(
        "outputs/part_b_tree_depth.csv",
        index=False
    )

    print("\nBest decision tree depth:",
          "unlimited" if best_depth is None else best_depth)

    models = {

        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "decision_tree": best_tree["model"],

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    validation_rmse = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        model_metrics = calculate_metrics(
            y_val,
            predictions
        )

        validation_rmse[model_name] = model_metrics["RMSE"]

        metrics_rows.append({
            "model": model_name,
            "split": "val",
            **model_metrics
        })

    learning_rate_rows = []

    for learning_rate in [0.01, 0.1, 1.0]:

        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=learning_rate,
            random_state=42
        )

        gb_model.fit(X_train, y_train)

        predictions = gb_model.predict(X_val)

        learning_rate_rows.append({
            "learning_rate": learning_rate,
            "val_rmse": round(
                float(np.sqrt(
                    mean_squared_error(y_val, predictions)
                )), 4
            ),
            "val_r2": round(
                float(r2_score(y_val, predictions)), 4
            )
        })

    pd.DataFrame(learning_rate_rows).to_csv(
        "outputs/part_b_gb_learning_rate.csv",
        index=False
    )

    # Rank four candidates using validation RMSE
    ranked_models = sorted(
        validation_rmse,
        key=validation_rmse.get
    )

    top_two_models = ranked_models[:2]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_rows = []

    for model_name in top_two_models:

        cv_scores = cross_val_score(
            models[model_name],
            X_train_val,
            y_train_val,
            cv=kfold
        )

        cv_rows.append({
            "model": model_name,
            "mean_r2": round(float(cv_scores.mean()), 4),
            "std_r2": round(float(cv_scores.std()), 4)
        })

    pd.DataFrame(cv_rows).to_csv(
        "outputs/part_b_cv.csv",
        index=False
    )

    final_model_name = ranked_models[0]

    final_model = models[final_model_name]

    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    metrics_rows.append({
        "model": final_model_name,
        "split": "test",
        **test_metrics
    })

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "model",
            "split",
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ]
    )

    numeric_columns = ["MAE", "MSE", "RMSE", "R2"]

    metrics_df[numeric_columns] = (
        metrics_df[numeric_columns].round(4)
    )

    metrics_df.to_csv(
        "outputs/part_b_metrics.csv",
        index=False
    )

    print("\n--- PART B CHECKS ---")
    print(checks.to_string(index=False))

    print("\n--- TREE DEPTH RESULTS ---")
    print(pd.DataFrame(tree_depth_rows).to_string(index=False))

    print("\n--- REGRESSION METRICS ---")
    print(metrics_df.to_string(index=False))

    print("\n--- LEARNING RATE RESULTS ---")
    print(pd.DataFrame(learning_rate_rows).to_string(index=False))

    print("\n--- CROSS VALIDATION ---")
    print(pd.DataFrame(cv_rows).to_string(index=False))

    print("\n--- FINAL MODEL ---")
    print("Selected model:", final_model_name)

    print("\nGenerated Part B files:")
    print("- outputs/part_b_checks.csv")
    print("- outputs/part_b_tree_depth.csv")
    print("- outputs/part_b_metrics.csv")
    print("- outputs/part_b_gb_learning_rate.csv")
    print("- outputs/part_b_cv.csv")


if __name__ == "__main__":
    main()
