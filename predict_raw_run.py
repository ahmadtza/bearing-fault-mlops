import sys

import h5py
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from feature_engineering import (
    SELECTED_FEATURES,
    SAMPLING_FREQUENCY,
    WINDOW_SIZE,
    STEP_SIZE,
    extract_features_from_recording
)


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = (
    "http://127.0.0.1:5000"
)

MODEL_URI = (
    "models:/bearing_condition_logistic@champion"
)

RAW_DATA_PATH = (
    "data/MachineData_export.mat"
)

RUN_THRESHOLD = 0.5


# ============================================================
# 2. Select Run ID
# ============================================================

if len(sys.argv) > 1:

    RUN_ID = int(
        sys.argv[1]
    )

else:

    RUN_ID = 1


if RUN_ID < 1 or RUN_ID > 40:

    raise ValueError(
        "RUN_ID must be between 1 and 40."
    )


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
print("RAW SIGNAL → CHAMPION MODEL INFERENCE")
print("=" * 80)


print(
    f"\nRun ID             : "
    f"{RUN_ID}"
)

print(
    f"Model URI          : "
    f"{MODEL_URI}"
)

print(
    f"Sampling frequency : "
    f"{SAMPLING_FREQUENCY} Hz"
)

print(
    f"Window size        : "
    f"{WINDOW_SIZE}"
)

print(
    f"Step size          : "
    f"{STEP_SIZE}"
)


# ============================================================
# 4. Load raw MATLAB export
# ============================================================

with h5py.File(
    RAW_DATA_PATH,
    "r"
) as file:

    ch1_all = np.array(
        file["ch1"]
    )

    ch2_all = np.array(
        file["ch2"]
    )

    ch3_all = np.array(
        file["ch3"]
    )


    run_ids = (
        np.array(
            file["run_id"]
        )
        .flatten()
        .astype(int)
    )


    labels = (
        np.array(
            file["label_code"]
        )
        .flatten()
        .astype(int)
    )


# ============================================================
# 5. Find requested run
# ============================================================

matching_indices = np.where(
    run_ids == RUN_ID
)[0]


if len(
    matching_indices
) == 0:

    raise ValueError(
        f"Run {RUN_ID} was not found."
    )


recording_index = int(
    matching_indices[0]
)


# ============================================================
# 6. Extract raw recording
# ============================================================

ch1 = ch1_all[
    recording_index
]

ch2 = ch2_all[
    recording_index
]

ch3 = ch3_all[
    recording_index
]


true_label = int(
    labels[
        recording_index
    ]
)


print("\n" + "=" * 80)
print("RAW RECORDING")
print("=" * 80)


print(
    f"Samples ch1 : "
    f"{len(ch1)}"
)

print(
    f"Samples ch2 : "
    f"{len(ch2)}"
)

print(
    f"Samples ch3 : "
    f"{len(ch3)}"
)


# ============================================================
# 7. Data quality check
# ============================================================

for name, signal in [
    ("ch1", ch1),
    ("ch2", ch2),
    ("ch3", ch3)
]:

    if np.isnan(signal).any():

        raise ValueError(
            f"{name} contains NaN values."
        )


    if np.isinf(signal).any():

        raise ValueError(
            f"{name} contains Inf values."
        )


print(
    "Raw signal quality check: PASS"
)


# ============================================================
# 8. Feature extraction
# ============================================================

print("\nExtracting frequency features...")


feature_df = (
    extract_features_from_recording(
        ch1,
        ch2,
        ch3
    )
)


print(
    "Feature extraction completed."
)


print(
    f"Number of windows: "
    f"{len(feature_df)}"
)


# ============================================================
# 9. Validate feature contract
# ============================================================

missing_features = [
    feature
    for feature in SELECTED_FEATURES
    if feature not in feature_df.columns
]


if missing_features:

    raise RuntimeError(
        f"Missing features: "
        f"{missing_features}"
    )


X = feature_df[
    SELECTED_FEATURES
]


print("\n" + "=" * 80)
print("FEATURE CONTRACT")
print("=" * 80)


print(
    f"Expected features : "
    f"{len(SELECTED_FEATURES)}"
)

print(
    f"Generated features: "
    f"{X.shape[1]}"
)

print(
    "Feature contract  : PASS"
)


# ============================================================
# 10. Load Champion
# ============================================================

print("\nLoading MLflow Champion...")


model = mlflow.sklearn.load_model(
    MODEL_URI
)


print(
    "Champion loaded successfully."
)


# ============================================================
# 11. Window predictions
# ============================================================

window_predictions = model.predict(
    X
)


window_probabilities = (
    model.predict_proba(
        X
    )[:, 1]
)


# ============================================================
# 12. Store window results
# ============================================================

result_df = pd.DataFrame(
    {
        "window_id":
            feature_df[
                "window_id"
            ],

        "start_sample":
            feature_df[
                "start_sample"
            ],

        "end_sample":
            feature_df[
                "end_sample"
            ],

        "anomaly_probability":
            window_probabilities,

        "prediction":
            window_predictions
    }
)


# ============================================================
# 13. Run-level aggregation
# ============================================================

mean_probability = float(
    np.mean(
        window_probabilities
    )
)


run_prediction = int(
    mean_probability
    >= RUN_THRESHOLD
)


# ============================================================
# 14. Engineering labels
# ============================================================

if run_prediction == 1:

    predicted_label = "Before"
    predicted_condition = "Anomalous"

else:

    predicted_label = "After"
    predicted_condition = "Normal"


if true_label == 1:

    true_name = "Before"
    true_condition = "Anomalous"

else:

    true_name = "After"
    true_condition = "Normal"


# ============================================================
# 15. Window statistics
# ============================================================

normal_windows = int(
    np.sum(
        window_predictions == 0
    )
)


anomalous_windows = int(
    np.sum(
        window_predictions == 1
    )
)


# ============================================================
# 16. Display sample predictions
# ============================================================

print("\n" + "=" * 80)
print("WINDOW PREDICTION SUMMARY")
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


print("\nFirst 10 windows:")


print(
    result_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. Final Run prediction
# ============================================================

print("\n" + "=" * 80)
print("FINAL RUN DIAGNOSIS")
print("=" * 80)


print(
    f"Mean anomaly probability : "
    f"{mean_probability:.6f}"
)

print(
    f"Decision threshold       : "
    f"{RUN_THRESHOLD:.6f}"
)

print(
    f"Predicted label          : "
    f"{predicted_label}"
)

print(
    f"Predicted condition      : "
    f"{predicted_condition}"
)


# ============================================================
# 18. Ground truth
# ============================================================

print("\n" + "=" * 80)
print("GROUND TRUTH CHECK")
print("=" * 80)


print(
    f"True label      : "
    f"{true_name}"
)

print(
    f"True condition  : "
    f"{true_condition}"
)

print(
    f"Correct         : "
    f"{run_prediction == true_label}"
)


print("\n" + "=" * 80)
print("END-TO-END RAW SIGNAL INFERENCE COMPLETED")
print("=" * 80)