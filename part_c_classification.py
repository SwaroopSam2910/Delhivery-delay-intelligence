
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid
)

from src.shipment_prep import FEATURES , NOISE_COLUMNS

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.feature_selection import SelectKBest, f_classif

def calculate_classification_metrics(y_true, y_pred):

    """
    Calculate confusion matrix and classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )
    }

def main():

    data = pd.read_csv("data/modelling_table.csv")

    X = data[FEATURES].copy()
    y = data["is_delayed"].copy()

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

    print("\n========== PART C: DATA SPLIT ==========")

    print("\nTraining:")
    print("Total rows:", len(X_train))
    print("Delayed:", int(y_train.sum()))
    print("On-time:", int((y_train == 0).sum()))

    print("\nValidation:")
    print("Total rows:", len(X_val))
    print("Delayed:", int(y_val.sum()))
    print("On-time:", int((y_val == 0).sum()))

    print("\nTest:")
    print("Total rows:", len(X_test))
    print("Delayed:", int(y_test.sum()))
    print("On-time:", int((y_test == 0).sum()))

    assert len(X_train) == 840
    assert len(X_val) == 180
    assert len(X_test) == 180

    assert int(y_train.sum()) == 195
    assert int(y_val.sum()) == 42
    assert int(y_test.sum()) == 41

    print("\nAll stratified split checks passed!")

    baseline_predictions = np.zeros(
        len(y_val),
        dtype=int
    )

    baseline_metrics = calculate_classification_metrics(
        y_val,
        baseline_predictions
    )

    models_rows = []

    models_rows.append({
        "model": "blind_all_on_time",
        **baseline_metrics
    })

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_val)

    logistic_metrics = calculate_classification_metrics(
        y_val,
        logistic_predictions
    )

    models_rows.append({
        "model": "logistic_regression",
        **logistic_metrics
    })

    tree_depths = [2, 3, 4, 5, 6, 8, 10]

    tree_candidates = []

    for depth in tree_depths:

        tree_model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=42
        )

        tree_model.fit(X_train, y_train)

        predictions = tree_model.predict(X_val)

        tree_f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        tree_candidates.append({
            "depth": depth,
            "f1": tree_f1,
            "model": tree_model
        })

    best_tree = min(
        tree_candidates,
        key=lambda item: (-item["f1"], item["depth"])
    )

    best_tree_depth = best_tree["depth"]
    best_tree_model = best_tree["model"]

    best_tree_predictions = best_tree_model.predict(X_val)

    best_tree_metrics = calculate_classification_metrics(
        y_val,
        best_tree_predictions
    )

    models_rows.append({
        "model": f"decision_tree_depth_{best_tree_depth}",
        **best_tree_metrics
    })

    print("\nBest Decision Tree Depth:", best_tree_depth)

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    rf_predictions = random_forest.predict(X_val)

    rf_metrics = calculate_classification_metrics(
        y_val,
        rf_predictions
    )

    models_rows.append({
        "model": "random_forest_default",
        **rf_metrics
    })

    models_df = pd.DataFrame(
        models_rows,
        columns=[
            "model",
            "TN",
            "FP",
            "FN",
            "TP",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    models_df[metric_columns] = (
        models_df[metric_columns].round(4)
    )

    models_df.to_csv(
        "outputs/part_c_models.csv",
        index=False
    )

    print("\n========== CLASSIFICATION RESULTS ==========")

    print(models_df.to_string(index=False))

    print("\nGenerated file:")
    print("outputs/part_c_models.csv")

    
    rf_importances = random_forest.feature_importances_

    f_selector = SelectKBest(
        score_func=f_classif,
        k="all"
    )

    f_selector.fit(X_train, y_train)

    f_scores = f_selector.scores_

    f_ranks = pd.Series(f_scores).rank(
        ascending=False,
        method="min"
    ).astype(int).values

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "f_rank": f_ranks,
        "rf_importance": rf_importances,
        "is_noise": [
            feature in NOISE_COLUMNS
            for feature in FEATURES
        ]
    })

    feature_scores_df[
        ["f_score", "rf_importance"]
    ] = feature_scores_df[
        ["f_score", "rf_importance"]
    ].round(4)

    feature_scores_df.to_csv(
        "outputs/part_c_feature_scores.csv",
        index=False
    )

    noise_importance = feature_scores_df.loc[
        feature_scores_df["is_noise"],
        "rf_importance"
    ].sum()

    print("\n========== FEATURE IMPORTANCE ==========")

    print(
        feature_scores_df.sort_values(
            "f_rank"
        ).to_string(index=False)
    )

    print("\nCombined noise importance:",
          round(float(noise_importance), 4))

    k_values = [5, 7, 10, 12, 17]

    selectk_rows = []

    for k in k_values:

        selection_pipeline = Pipeline([
            (
                "selector",
                SelectKBest(
                    score_func=f_classif,
                    k=k
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ])

        selection_pipeline.fit(X_train, y_train)

        predictions = selection_pipeline.predict(X_val)

        val_f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        selectk_rows.append({
            "k": k,
            "val_f1": round(float(val_f1), 4)
        })

    selectk_df = pd.DataFrame(selectk_rows)

    selectk_df.to_csv(
        "outputs/part_c_selectk.csv",
        index=False
    )

    print("\n========== SELECTKBEST RESULTS ==========")

    print(selectk_df.to_string(index=False))

    print("\nGenerated files:")
    print("- outputs/part_c_feature_scores.csv")
    print("- outputs/part_c_selectk.csv")

    selector_7 = SelectKBest(
        score_func=f_classif,
        k=7
    )

    X_train_selected = selector_7.fit_transform(
        X_train,
        y_train
    )

    X_val_selected = selector_7.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid=param_grid,
        cv=5
    )

    grid_search.fit(
        X_train_selected,
        y_train
    )

    grid_predictions = grid_search.best_estimator_.predict(
        X_val_selected
    )

    grid_val_f1 = f1_score(
        y_val,
        grid_predictions,
        zero_division=0
    )

    grid_candidates = len(
        list(ParameterGrid(param_grid))
    )

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_distributions=param_grid,
        n_iter=9,
        cv=5,
        random_state=42
    )

    random_search.fit(
        X_train_selected,
        y_train
    )

    random_predictions = random_search.best_estimator_.predict(
        X_val_selected
    )

    random_val_f1 = f1_score(
        y_val,
        random_predictions,
        zero_division=0
    )

    random_candidates = len(
        random_search.cv_results_["params"]
    )

    default_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1
    )

    default_forest.fit(
        X_train_selected,
        y_train
    )

    default_predictions = default_forest.predict(
        X_val_selected
    )

    default_val_f1 = f1_score(
        y_val,
        default_predictions,
        zero_division=0
    )

    tuning_rows = [

        {
            "search": "grid_search",
            "candidates": grid_candidates,
            "cv_folds": 5,
            "total_fits": grid_candidates * 5,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(
                float(grid_search.best_score_), 4
            ),
            "val_f1": round(float(grid_val_f1), 4)
        },

        {
            "search": "randomized_search",
            "candidates": random_candidates,
            "cv_folds": 5,
            "total_fits": random_candidates * 5,
            "best_params": str(random_search.best_params_),
            "best_cv_accuracy": round(
                float(random_search.best_score_), 4
            ),
            "val_f1": round(float(random_val_f1), 4)
        },

        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": round(float(default_val_f1), 4)
        }
    ]

    tuning_df = pd.DataFrame(
        tuning_rows,
        columns=[
            "search",
            "candidates",
            "cv_folds",
            "total_fits",
            "best_params",
            "best_cv_accuracy",
            "val_f1"
        ]
    )

    tuning_df.to_csv(
        "outputs/part_c_tuning.csv",
        index=False
    )

    candidate_configs = [

        {
            "name": "grid_search",
            "params": grid_search.best_params_,
            "val_f1": grid_val_f1
        },

        {
            "name": "randomized_search",
            "params": random_search.best_params_,
            "val_f1": random_val_f1
        },

        {
            "name": "default_forest_same_7_features",
            "params": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_leaf": 1
            },
            "val_f1": default_val_f1
        }
    ]

    priority = {
        "default_forest_same_7_features": 0,
        "grid_search": 1,
        "randomized_search": 2
    }

    chosen_candidate = min(
        candidate_configs,
        key=lambda candidate: (
            -candidate["val_f1"],
            priority[candidate["name"]]
        )
    )

    chosen_from = chosen_candidate["name"]
    chosen_params = chosen_candidate["params"]

    print("\n========== HYPERPARAMETER TUNING ==========")

    print(tuning_df.to_string(index=False))

    print("\nChosen configuration:", chosen_from)
    print("Chosen parameters:", chosen_params)

    print("\nGenerated file:")
    print("outputs/part_c_tuning.csv")

    
    # --------------------------------------------------
    # 22. BUILD FINAL CLASSIFICATION PIPELINE
    # --------------------------------------------------

    # Use the selected seven features and winning parameters
    final_model = Pipeline([
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=7
            )
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **chosen_params
            )
        )
    ])

    # Fit using training data only
    final_model.fit(X_train, y_train)

    # --------------------------------------------------
    # 23. VALIDATION PROBABILITIES
    # --------------------------------------------------

    # Probability of the positive class (delayed = 1)
    val_probabilities = final_model.predict_proba(X_val)[:, 1]

    # Exact threshold list required by the brief
    thresholds = [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9
    ]

    threshold_rows = []

    for threshold in thresholds:

        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        threshold_metrics = calculate_classification_metrics(
            y_val,
            predictions
        )

        threshold_rows.append({
            "threshold": threshold,
            **threshold_metrics
        })

    threshold_df = pd.DataFrame(
        threshold_rows,
        columns=[
            "threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    eligible_thresholds = [
        row["threshold"]
        for row in threshold_rows
        if row["recall"] >= 0.80
    ]

    if eligible_thresholds:
        chosen_threshold = max(eligible_thresholds)
    else:
        chosen_threshold = 0.1

    print("\n========== ALERT THRESHOLD ==========")

    print(
        threshold_df.round(4).to_string(index=False)
    )

    print("\nChosen alert threshold:", chosen_threshold)

    threshold_output = threshold_df.copy()

    threshold_output[
        ["accuracy", "precision", "recall", "f1"]
    ] = threshold_output[
        ["accuracy", "precision", "recall", "f1"]
    ].round(4)

    threshold_output.to_csv(
        "outputs/part_c_thresholds.csv",
        index=False
    )

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= chosen_threshold
    ).astype(int)

    test_metrics = calculate_classification_metrics(
        y_test,
        test_predictions
    )

    final_test_row = {
        "threshold": chosen_threshold,
        **test_metrics
    }

    final_test_df = pd.DataFrame([final_test_row])

    final_test_df[
        ["accuracy", "precision", "recall", "f1"]
    ] = final_test_df[
        ["accuracy", "precision", "recall", "f1"]
    ].round(4)

    final_test_df.to_csv(
        "outputs/part_c_final_test.csv",
        index=False
    )

    max_depth_value = chosen_params["max_depth"]

    final_config = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": (
            "unlimited"
            if max_depth_value is None
            else max_depth_value
        ),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": chosen_from
    }])

    final_config.to_csv(
        "outputs/part_c_final_config.csv",
        index=False
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(final_test_df.to_string(index=False))

    print("\n========== FINAL MODEL CONFIGURATION ==========")
    print(final_config.to_string(index=False))

    print("\nGenerated Part C files:")
    print("- outputs/part_c_thresholds.csv")
    print("- outputs/part_c_final_test.csv")
    print("- outputs/part_c_final_config.csv")

if __name__ == "__main__":
    main()
