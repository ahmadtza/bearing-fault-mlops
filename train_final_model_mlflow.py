import os
import json
import tempfile

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = "data/grouped_time_frequency_hz_features.csv"

EXPERIMENT_NAME = "Bearing Fault Detection"

MODEL_ARTIFACT_PATH = "model"

RANDOM_STATE = 42


# ============================================================
# 2. Signal-processing configuration
# ============================================================

SAMPLING_FREQUENCY = 12000
WINDOW_SIZE = 2048
STEP_SIZE = 1024

WINDOW_OVERLAP = WINDOW_SIZE - STEP_SIZE

WINDOW_OVERLAP_PERCENT = (
    100 * WINDOW_OVERLAP / WINDOW_SIZE
)


# ============================================================
# 3. Selected frequency features
# ============================================================

SELECTED_FEATURES = [
    "ch3_SpectralCentroidHz",
    "ch3_BandEnergy_1000_2000",
    "ch3_RelEnergy_2000_4000",
    "ch3_SpectralFlatness",
    "ch3_BandEnergy_100_500",
    "ch3_RelEnergy_100_500",
    "ch3_BandEnergy_500_1000",
    "ch3_BandEnergy_2000_4000",
    "ch1_SpectralFlatness",
    "ch3_SpectralSpreadHz",
]


# ============================================================
# 4. Previously established group-aware CV metrics
#
# These metrics come from the completed 5-fold
# StratifiedGroupKFold experiment.
# They are NOT calculated from the final training set.
# ============================================================

CV_RUN_METRICS = {
    "cv_run_accuracy_mean": 1.0,
    "cv_run_precision_mean": 1.0,
    "cv_run_recall_mean": 1.0,
    "cv_run_f1_mean": 1.0,
    "cv_run_roc_auc_mean": 1.0,
    "cv_run_pr_auc_mean": 1.0,
}


# ============================================================
# 5. Load dataset
# ============================================================

print("=" * 80)
print("MLFLOW FINAL MODEL TRAINING")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nNumber of runs:")
print(df["run_id"].nunique())

print("\nClass distribution:")
print(
    df.groupby("run_id")["label"]
    .first()
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. Validate required columns
# ============================================================

required_columns = (
    ["run_id", "label"]
    + SELECTED_FEATURES
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 7. Prepare X and y
# ============================================================

X = df[SELECTED_FEATURES].copy()

y = df["label"].astype(int).copy()


# ============================================================
# 8. Create final sklearn pipeline
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        ),
    ]
)


# ============================================================
# 9. Configure MLflow
# ============================================================
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# 10. Start MLflow run
# ============================================================

with mlflow.start_run(
    run_name="final_logistic_frequency_model"
) as run:

    print("\n" + "=" * 80)
    print("MLFLOW RUN STARTED")
    print("=" * 80)

    print(f"Run ID: {run.info.run_id}")


    # ========================================================
    # 11. Log model / signal-processing parameters
    # ========================================================

    mlflow.log_params(
        {
            "model_type":
                "LogisticRegression",

            "random_state":
                RANDOM_STATE,

            "max_iter":
                3000,

            "sampling_frequency_hz":
                SAMPLING_FREQUENCY,

            "window_size_samples":
                WINDOW_SIZE,

            "step_size_samples":
                STEP_SIZE,

            "window_overlap_samples":
                WINDOW_OVERLAP,

            "window_overlap_percent":
                WINDOW_OVERLAP_PERCENT,

            "number_of_features":
                len(SELECTED_FEATURES),

            "number_of_windows":
                len(df),

            "number_of_runs":
                df["run_id"].nunique(),

            "run_aggregation":
                "mean_window_probability",

            "run_threshold":
                0.5,

            "validation_strategy":
                "5_fold_stratified_group_kfold",
        }
    )


    # ========================================================
    # 12. Train final model on all available windows
    # ========================================================

    pipeline.fit(
        X,
        y
    )


    # ========================================================
    # 13. Window-level training sanity check
    # ========================================================

    window_predictions = pipeline.predict(
        X
    )

    window_probabilities = (
        pipeline.predict_proba(X)[:, 1]
    )


    window_accuracy = accuracy_score(
        y,
        window_predictions
    )

    window_precision = precision_score(
        y,
        window_predictions
    )

    window_recall = recall_score(
        y,
        window_predictions
    )

    window_f1 = f1_score(
        y,
        window_predictions
    )


    print("\n" + "=" * 80)
    print("WINDOW-LEVEL TRAINING SANITY CHECK")
    print("=" * 80)

    print(
        f"Accuracy : {window_accuracy:.6f}"
    )

    print(
        f"Precision: {window_precision:.6f}"
    )

    print(
        f"Recall   : {window_recall:.6f}"
    )

    print(
        f"F1       : {window_f1:.6f}"
    )


    mlflow.log_metrics(
        {
            "train_window_accuracy":
                window_accuracy,

            "train_window_precision":
                window_precision,

            "train_window_recall":
                window_recall,

            "train_window_f1":
                window_f1,
        }
    )


    # ========================================================
    # 14. Run-level training sanity check
    # ========================================================

    run_results = pd.DataFrame(
        {
            "run_id":
                df["run_id"].values,

            "true_label":
                y.values,

            "window_probability":
                window_probabilities,
        }
    )


    run_summary = (
        run_results
        .groupby("run_id")
        .agg(
            true_label=(
                "true_label",
                "first"
            ),
            mean_probability=(
                "window_probability",
                "mean"
            ),
        )
        .reset_index()
    )


    run_summary["prediction"] = (
        run_summary["mean_probability"]
        >= 0.5
    ).astype(int)


    run_accuracy = accuracy_score(
        run_summary["true_label"],
        run_summary["prediction"]
    )

    run_precision = precision_score(
        run_summary["true_label"],
        run_summary["prediction"]
    )

    run_recall = recall_score(
        run_summary["true_label"],
        run_summary["prediction"]
    )

    run_f1 = f1_score(
        run_summary["true_label"],
        run_summary["prediction"]
    )


    print("\n" + "=" * 80)
    print("RUN-LEVEL TRAINING SANITY CHECK")
    print("=" * 80)

    print(
        f"Accuracy : {run_accuracy:.6f}"
    )

    print(
        f"Precision: {run_precision:.6f}"
    )

    print(
        f"Recall   : {run_recall:.6f}"
    )

    print(
        f"F1       : {run_f1:.6f}"
    )


    mlflow.log_metrics(
        {
            "train_run_accuracy":
                run_accuracy,

            "train_run_precision":
                run_precision,

            "train_run_recall":
                run_recall,

            "train_run_f1":
                run_f1,
        }
    )


    # ========================================================
    # 15. Log previously established CV metrics
    # ========================================================

    mlflow.log_metrics(
        CV_RUN_METRICS
    )


    # ========================================================
    # 16. Create metadata artifacts
    # ========================================================

    metadata = {
        "model_name":
            "Bearing Condition Logistic Regression",

        "model_type":
            "LogisticRegression",

        "problem_type":
            "binary_classification",

        "negative_class":
            {
                "code": 0,
                "label": "After",
                "meaning": "Normal",
            },

        "positive_class":
            {
                "code": 1,
                "label": "Before",
                "meaning": "Anomalous",
            },

        "sampling_frequency_hz":
            SAMPLING_FREQUENCY,

        "window_size_samples":
            WINDOW_SIZE,

        "step_size_samples":
            STEP_SIZE,

        "window_overlap_percent":
            WINDOW_OVERLAP_PERCENT,

        "number_of_features":
            len(SELECTED_FEATURES),

        "selected_features":
            SELECTED_FEATURES,

        "run_level_aggregation":
            "mean_window_probability",

        "run_level_threshold":
            0.5,

        "validation_protocol":
            "5-fold StratifiedGroupKFold using run_id",

        "cv_run_metrics":
            CV_RUN_METRICS,
    }


    # ========================================================
    # 17. Log JSON and CSV artifacts
    # ========================================================

    with tempfile.TemporaryDirectory() as temp_dir:

        features_path = os.path.join(
            temp_dir,
            "selected_features.json"
        )

        metadata_path = os.path.join(
            temp_dir,
            "model_metadata.json"
        )

        run_results_path = os.path.join(
            temp_dir,
            "run_training_predictions.csv"
        )


        with open(
            features_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                SELECTED_FEATURES,
                file,
                indent=4
            )


        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )


        run_summary.to_csv(
            run_results_path,
            index=False
        )


        mlflow.log_artifact(
            features_path,
            artifact_path="metadata"
        )

        mlflow.log_artifact(
            metadata_path,
            artifact_path="metadata"
        )

        mlflow.log_artifact(
            run_results_path,
            artifact_path="evaluation"
        )


    # ========================================================
    # 18. Create model signature
    # ========================================================

    signature = mlflow.models.infer_signature(
        X,
        pipeline.predict_proba(X)
    )


    # ========================================================
    # 19. Log sklearn pipeline
    # ========================================================

    input_example = X.iloc[
        :5
    ].copy()


    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,

        name=MODEL_ARTIFACT_PATH,

        signature=signature,

        input_example=input_example,
    )


    # ========================================================
    # 20. Add MLflow tags
    # ========================================================

    mlflow.set_tags(
        {
            "project":
                "bearing_fault_detection",

            "model_stage":
                "final_candidate",

            "feature_domain":
                "frequency",

            "decision_level":
                "run",

            "positive_class":
                "Before_Anomalous",

            "negative_class":
                "After_Normal",
        }
    )


    # ========================================================
    # 21. Final information
    # ========================================================

    print("\n" + "=" * 80)
    print("MLFLOW MODEL LOGGED SUCCESSFULLY")
    print("=" * 80)

    print(
        f"Run ID     : {run.info.run_id}"
    )

    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Model URI  : {model_info.model_uri}"
    )


print("\n" + "=" * 80)
print("MLFLOW TRAINING COMPLETED")
print("=" * 80)