import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. Configuration
# ============================================================

data_path = "data/grouped_time_frequency_hz_features.csv"

model_directory = "models"

model_path = os.path.join(
    model_directory,
    "bearing_logistic_pipeline.joblib"
)

metadata_path = os.path.join(
    model_directory,
    "model_metadata.json"
)

feature_list_path = os.path.join(
    model_directory,
    "selected_features.json"
)


random_state = 42


# ============================================================
# 2. Signal-processing configuration
# ============================================================

sampling_frequency = 12000

window_size = 2048

step_size = 1024


# ============================================================
# 3. Final selected frequency features
# ============================================================

selected_features = [
    "ch3_SpectralCentroidHz",
    "ch3_BandEnergy_1000_2000",
    "ch3_RelEnergy_2000_4000",
    "ch3_SpectralFlatness",
    "ch3_BandEnergy_100_500",
    "ch3_RelEnergy_100_500",
    "ch3_BandEnergy_500_1000",
    "ch3_BandEnergy_2000_4000",
    "ch1_SpectralFlatness",
    "ch3_SpectralSpreadHz"
]


# ============================================================
# 4. Create output directory
# ============================================================

os.makedirs(
    model_directory,
    exist_ok=True
)


# ============================================================
# 5. Load feature dataset
# ============================================================

df = pd.read_csv(
    data_path
)


print("=" * 80)
print("FINAL MODEL TRAINING")
print("=" * 80)


print("\nDataset shape:")
print(
    df.shape
)


print("\nSelected features:")

for feature in selected_features:
    print(
        " -",
        feature
    )


# ============================================================
# 6. Validate required columns
# ============================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "Missing required features: "
        + str(missing_features)
    )


# ============================================================
# 7. Define X and y
# ============================================================

X = df[
    selected_features
]


# 0 = After  = Normal
# 1 = Before = Anomalous

y = df[
    "label"
]


# ============================================================
# 8. Build final sklearn Pipeline
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
                random_state=random_state
            )
        )
    ]
)


# ============================================================
# 9. Train final model
#
# IMPORTANT:
# Cross-validation has already been used for unbiased
# performance estimation.
#
# Now the production model is trained using all available
# feature windows.
# ============================================================

pipeline.fit(
    X,
    y
)


# ============================================================
# 10. Training predictions
#
# These metrics are NOT final validation metrics.
# They are only a sanity check that the final fitted
# pipeline is functioning correctly.
# ============================================================

training_prediction = pipeline.predict(
    X
)


training_probability = (
    pipeline.predict_proba(
        X
    )[:, 1]
)


training_accuracy = accuracy_score(
    y,
    training_prediction
)

training_precision = precision_score(
    y,
    training_prediction
)

training_recall = recall_score(
    y,
    training_prediction
)

training_f1 = f1_score(
    y,
    training_prediction
)


print("\n" + "=" * 80)
print("FINAL TRAINING SANITY-CHECK METRICS")
print("=" * 80)


print(
    f"Accuracy : "
    f"{training_accuracy:.6f}"
)

print(
    f"Precision: "
    f"{training_precision:.6f}"
)

print(
    f"Recall   : "
    f"{training_recall:.6f}"
)

print(
    f"F1       : "
    f"{training_f1:.6f}"
)

# ============================================================
# 10B. Run-level sanity check
# ============================================================

run_check_df = pd.DataFrame(
    {
        "run_id": df["run_id"].values,
        "true_label": y.values,
        "window_probability": training_probability
    }
)


run_check_summary = (
    run_check_df
    .groupby("run_id")
    .agg(
        true_label=(
            "true_label",
            "first"
        ),

        mean_probability=(
            "window_probability",
            "mean"
        )
    )
    .reset_index()
)


run_check_summary["prediction"] = (
    run_check_summary["mean_probability"]
    >= 0.5
).astype(int)


run_accuracy = accuracy_score(
    run_check_summary["true_label"],
    run_check_summary["prediction"]
)

run_precision = precision_score(
    run_check_summary["true_label"],
    run_check_summary["prediction"]
)

run_recall = recall_score(
    run_check_summary["true_label"],
    run_check_summary["prediction"]
)

run_f1 = f1_score(
    run_check_summary["true_label"],
    run_check_summary["prediction"]
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
# ============================================================
# 11. Save complete sklearn pipeline
# ============================================================

joblib.dump(
    pipeline,
    model_path
)


# ============================================================
# 12. Save selected feature list
# ============================================================

with open(
    feature_list_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        selected_features,
        file,
        indent=4
    )


# ============================================================
# 13. Model metadata
# ============================================================

metadata = {

    "model_name":
        "Bearing Condition Logistic Regression",

    "model_type":
        "LogisticRegression",

    "problem_type":
        "Binary classification",

    "positive_class":
        {
            "code": 1,
            "label": "Before",
            "meaning": "Anomalous"
        },

    "negative_class":
        {
            "code": 0,
            "label": "After",
            "meaning": "Normal"
        },

    "number_of_features":
        len(selected_features),

    "selected_features":
        selected_features,

    "sampling_frequency_hz":
        sampling_frequency,

    "window_size_samples":
        window_size,

    "step_size_samples":
        step_size,

    "window_overlap_samples":
        window_size - step_size,

    "window_overlap_percent":
        100 * (
            window_size - step_size
        ) / window_size,

    "run_level_aggregation":
        "mean window probability",

    "run_level_threshold":
        0.5,

    "validation_protocol":
        "5-fold StratifiedGroupKFold using run_id",

    "final_cross_validation_results":
        {
            "run_level_accuracy_mean": 1.0,
            "run_level_precision_mean": 1.0,
            "run_level_recall_mean": 1.0,
            "run_level_f1_mean": 1.0,
            "run_level_roc_auc_mean": 1.0,
            "run_level_pr_auc_mean": 1.0
        },

    "training_dataset":
        {
            "number_of_runs": 40,
            "number_of_windows": int(
                len(df)
            ),
            "before_runs": 20,
            "after_runs": 20
        }
}


# ============================================================
# 14. Save metadata
# ============================================================

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


# ============================================================
# 15. Verify saved model
# ============================================================

loaded_pipeline = joblib.load(
    model_path
)


example_features = X.iloc[
    [0]
]


example_prediction = loaded_pipeline.predict(
    example_features
)[0]


example_probability = (
    loaded_pipeline.predict_proba(
        example_features
    )[0, 1]
)


# ============================================================
# 16. Display verification
# ============================================================

print("\n" + "=" * 80)
print("MODEL SERIALIZATION TEST")
print("=" * 80)


print(
    f"Example true label : "
    f"{int(y.iloc[0])}"
)

print(
    f"Example prediction : "
    f"{int(example_prediction)}"
)

print(
    f"Anomaly probability: "
    f"{example_probability:.6f}"
)


# ============================================================
# 17. Final output paths
# ============================================================

print("\n" + "=" * 80)
print("ARTIFACTS SAVED")
print("=" * 80)


print(
    f"Model pipeline : "
    f"{model_path}"
)

print(
    f"Feature list   : "
    f"{feature_list_path}"
)

print(
    f"Metadata       : "
    f"{metadata_path}"
)


print("\n" + "=" * 80)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 80)