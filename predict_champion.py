import sys
import json

import pandas as pd
import mlflow
import mlflow.sklearn


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = "http://127.0.0.1:5000"

MODEL_URI = (
    "models:/bearing_condition_logistic@champion"
)

DATA_PATH = (
    "data/grouped_time_frequency_hz_features.csv"
)

METADATA_PATH = (
    "models/model_metadata.json"
)


# ============================================================
# 2. Select Run ID
#
# Usage:
# python predict_champion.py
#
# or:
# python predict_champion.py 25
# ============================================================

if len(sys.argv) > 1:
    RUN_ID = int(sys.argv[1])
else:
    RUN_ID = 1


# ============================================================
# 3. Configure MLflow
# ============================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)

mlflow.set_registry_uri(
    TRACKING_URI
)


print("=" * 80)
print("BEARING CHAMPION MODEL INFERENCE")
print("=" * 80)

print(
    f"\nModel URI : {MODEL_URI}"
)

print(
    f"Run ID    : {RUN_ID}"
)


# ============================================================
# 4. Load Champion from MLflow Registry
# ============================================================

print("\nLoading Champion model...")


model = mlflow.sklearn.load_model(
    MODEL_URI
)


print(
    "Champion model loaded successfully."
)


# ============================================================
# 5. Load model metadata
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(
        file
    )


selected_features = metadata[
    "selected_features"
]


run_threshold = metadata[
    "run_level_threshold"
]


print("\n" + "=" * 80)
print("MODEL CONTRACT")
print("=" * 80)

print(
    f"Number of features : "
    f"{len(selected_features)}"
)

print(
    f"Run threshold      : "
    f"{run_threshold}"
)

print(
    f"Sampling frequency : "
    f"{metadata['sampling_frequency_hz']} Hz"
)

print(
    f"Window size        : "
    f"{metadata['window_size_samples']}"
)

print(
    f"Step size          : "
    f"{metadata['step_size_samples']}"
)


# ============================================================
# 6. Load feature dataset
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 7. Select requested recording
# ============================================================

run_df = df[
    df["run_id"] == RUN_ID
].copy()


if run_df.empty:

    available_runs = sorted(
        df["run_id"]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Run ID {RUN_ID} does not exist. "
        f"Available runs: {available_runs}"
    )


print("\n" + "=" * 80)
print("RUN INFORMATION")
print("=" * 80)

print(
    f"Run ID            : {RUN_ID}"
)

print(
    f"Number of windows : {len(run_df)}"
)


# ============================================================
# 8. Check required features
# ============================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in run_df.columns
]


if missing_features:

    raise RuntimeError(
        f"Missing required features: "
        f"{missing_features}"
    )


X_run = run_df[
    selected_features
]


# ============================================================
# 9. Window-level prediction
# ============================================================

window_predictions = model.predict(
    X_run
)


window_probabilities = (
    model.predict_proba(
        X_run
    )[:, 1]
)


# ============================================================
# 10. Add predictions to DataFrame
# ============================================================

prediction_df = pd.DataFrame(
    {
        "window_id":
            run_df["window_id"].values,

        "anomaly_probability":
            window_probabilities,

        "prediction":
            window_predictions
    }
)


# ============================================================
# 11. Run-level aggregation
# ============================================================

mean_probability = (
    window_probabilities.mean()
)


final_prediction = int(
    mean_probability
    >= run_threshold
)


# ============================================================
# 12. Convert prediction to engineering meaning
# ============================================================

if final_prediction == 1:

    predicted_label = "Before"

    predicted_condition = (
        "Anomalous"
    )

else:

    predicted_label = "After"

    predicted_condition = (
        "Normal"
    )


# ============================================================
# 13. True label
#
# Only available because this is our labelled development
# dataset. A real production recording will not have it.
# ============================================================

true_code = int(
    run_df["label"].iloc[0]
)


if true_code == 1:

    true_label = "Before"

    true_condition = "Anomalous"

else:

    true_label = "After"

    true_condition = "Normal"


# ============================================================
# 14. Window statistics
# ============================================================

normal_windows = int(
    (window_predictions == 0).sum()
)


anomalous_windows = int(
    (window_predictions == 1).sum()
)


# ============================================================
# 15. Display final result
# ============================================================

print("\n" + "=" * 80)
print("WINDOW-LEVEL SUMMARY")
print("=" * 80)

print(
    f"Normal windows    : "
    f"{normal_windows}"
)

print(
    f"Anomalous windows : "
    f"{anomalous_windows}"
)

print(
    f"Total windows     : "
    f"{len(window_predictions)}"
)


print("\nFirst 10 window predictions:")

print(
    prediction_df
    .head(10)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("RUN-LEVEL PREDICTION")
print("=" * 80)

print(
    f"Mean anomaly probability : "
    f"{mean_probability:.6f}"
)

print(
    f"Decision threshold       : "
    f"{run_threshold:.6f}"
)

print(
    f"Predicted label          : "
    f"{predicted_label}"
)

print(
    f"Predicted condition      : "
    f"{predicted_condition}"
)


print("\n" + "=" * 80)
print("GROUND TRUTH CHECK")
print("=" * 80)

print(
    f"True label      : "
    f"{true_label}"
)

print(
    f"True condition  : "
    f"{true_condition}"
)

print(
    f"Correct         : "
    f"{final_prediction == true_code}"
)


print("\n" + "=" * 80)
print("CHAMPION INFERENCE COMPLETED")
print("=" * 80)