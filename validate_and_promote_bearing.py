import json
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "bearing_condition_logistic"

CANDIDATE_ALIAS = "candidate"

CHAMPION_ALIAS = "champion"

DATA_PATH = "data/grouped_time_frequency_hz_features.csv"

METADATA_PATH = "models/model_metadata.json"


# ============================================================
# 2. Promotion criteria
# ============================================================

MIN_RUN_ACCURACY = 0.95
MIN_RUN_PRECISION = 0.95
MIN_RUN_RECALL = 0.95
MIN_RUN_F1 = 0.95


# ============================================================
# 3. Configure MLflow
# ============================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)

mlflow.set_registry_uri(
    TRACKING_URI
)


client = MlflowClient(
    tracking_uri=TRACKING_URI,
    registry_uri=TRACKING_URI
)


print("=" * 80)
print("BEARING MODEL VALIDATION AND PROMOTION")
print("=" * 80)


# ============================================================
# 4. Get candidate model version
# ============================================================

candidate_version = (
    client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CANDIDATE_ALIAS
    )
)


candidate_version_number = str(
    candidate_version.version
)


print("\n" + "=" * 80)
print("CANDIDATE MODEL")
print("=" * 80)

print(
    f"Model   : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Version : "
    f"{candidate_version_number}"
)

print(
    f"Status  : "
    f"{candidate_version.status}"
)

print(
    f"Run ID  : "
    f"{candidate_version.run_id}"
)


# ============================================================
# 5. Candidate URI
# ============================================================

candidate_uri = (
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{CANDIDATE_ALIAS}"
)


print(
    f"URI     : "
    f"{candidate_uri}"
)


# ============================================================
# 6. Load model
# ============================================================

print("\nLoading candidate model...")


model = mlflow.sklearn.load_model(
    candidate_uri
)


print(
    "Candidate model loaded successfully."
)


# ============================================================
# 7. Load metadata contract
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


expected_number_of_features = metadata[
    "number_of_features"
]


run_threshold = metadata[
    "run_level_threshold"
]


print("\n" + "=" * 80)
print("MODEL CONTRACT")
print("=" * 80)

print(
    f"Expected features : "
    f"{expected_number_of_features}"
)

print(
    f"Run threshold     : "
    f"{run_threshold}"
)

print(
    f"Sampling frequency: "
    f"{metadata['sampling_frequency_hz']} Hz"
)

print(
    f"Window size       : "
    f"{metadata['window_size_samples']}"
)

print(
    f"Step size         : "
    f"{metadata['step_size_samples']}"
)


# ============================================================
# 8. Load validation dataset
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# 9. Validate feature contract
# ============================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]


feature_contract_passed = (
    len(missing_features) == 0
    and
    len(selected_features)
    == expected_number_of_features
)


print("\n" + "=" * 80)
print("FEATURE CONTRACT CHECK")
print("=" * 80)

print(
    f"Missing features: "
    f"{missing_features}"
)

print(
    f"Feature count    : "
    f"{len(selected_features)}"
)

print(
    f"Expected count   : "
    f"{expected_number_of_features}"
)

print(
    f"Status           : "
    f"{'PASS' if feature_contract_passed else 'FAIL'}"
)


# ============================================================
# 10. Prepare validation features
# ============================================================

if not feature_contract_passed:

    raise RuntimeError(
        "Feature contract validation failed."
    )


X = df[
    selected_features
]


y = df[
    "label"
].astype(int)


# ============================================================
# 11. Model inference check
# ============================================================

print("\n" + "=" * 80)
print("INFERENCE CHECK")
print("=" * 80)


try:

    test_prediction = model.predict(
        X.iloc[:5]
    )


    test_probability = (
        model.predict_proba(
            X.iloc[:5]
        )[:, 1]
    )


    inference_passed = True


    print(
        "Prediction test: PASS"
    )

    print(
        "Example predictions:"
    )

    print(
        test_prediction
    )

    print(
        "Example anomaly probabilities:"
    )

    print(
        test_probability
    )


except Exception as error:

    inference_passed = False

    print(
        "Prediction test: FAIL"
    )

    print(
        error
    )


# ============================================================
# 12. Window predictions
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
# 13. Window-level sanity metrics
# ============================================================

window_accuracy = accuracy_score(
    y,
    window_predictions
)

window_precision = precision_score(
    y,
    window_predictions,
    zero_division=0
)

window_recall = recall_score(
    y,
    window_predictions,
    zero_division=0
)

window_f1 = f1_score(
    y,
    window_predictions,
    zero_division=0
)


print("\n" + "=" * 80)
print("WINDOW-LEVEL SANITY METRICS")
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


# ============================================================
# 14. Run-level aggregation
# ============================================================

run_df = pd.DataFrame(
    {
        "run_id":
            df["run_id"].values,

        "true_label":
            y.values,

        "window_probability":
            window_probabilities
    }
)


run_summary = (
    run_df
    .groupby(
        "run_id"
    )
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


run_summary[
    "prediction"
] = (
    run_summary[
        "mean_probability"
    ]
    >= run_threshold
).astype(int)


# ============================================================
# 15. Run-level metrics
# ============================================================

run_accuracy = accuracy_score(
    run_summary[
        "true_label"
    ],
    run_summary[
        "prediction"
    ]
)

run_precision = precision_score(
    run_summary[
        "true_label"
    ],
    run_summary[
        "prediction"
    ],
    zero_division=0
)

run_recall = recall_score(
    run_summary[
        "true_label"
    ],
    run_summary[
        "prediction"
    ],
    zero_division=0
)

run_f1 = f1_score(
    run_summary[
        "true_label"
    ],
    run_summary[
        "prediction"
    ],
    zero_division=0
)


print("\n" + "=" * 80)
print("RUN-LEVEL VALIDATION METRICS")
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
# 16. Promotion criteria
# ============================================================

accuracy_passed = (
    run_accuracy
    >= MIN_RUN_ACCURACY
)

precision_passed = (
    run_precision
    >= MIN_RUN_PRECISION
)

recall_passed = (
    run_recall
    >= MIN_RUN_RECALL
)

f1_passed = (
    run_f1
    >= MIN_RUN_F1
)


promotion_passed = all(
    [
        feature_contract_passed,
        inference_passed,
        accuracy_passed,
        precision_passed,
        recall_passed,
        f1_passed
    ]
)


# ============================================================
# 17. Validation report
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION GATE")
print("=" * 80)


print(
    f"Feature contract : "
    f"{'PASS' if feature_contract_passed else 'FAIL'}"
)

print(
    f"Inference check  : "
    f"{'PASS' if inference_passed else 'FAIL'}"
)

print(
    f"Accuracy >= {MIN_RUN_ACCURACY:.2f}: "
    f"{'PASS' if accuracy_passed else 'FAIL'}"
)

print(
    f"Precision >= {MIN_RUN_PRECISION:.2f}: "
    f"{'PASS' if precision_passed else 'FAIL'}"
)

print(
    f"Recall >= {MIN_RUN_RECALL:.2f}: "
    f"{'PASS' if recall_passed else 'FAIL'}"
)

print(
    f"F1 >= {MIN_RUN_F1:.2f}: "
    f"{'PASS' if f1_passed else 'FAIL'}"
)


# ============================================================
# 18. Promotion
# ============================================================

if promotion_passed:

    print("\n" + "=" * 80)
    print("PROMOTION PASSED")
    print("=" * 80)


    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=candidate_version_number
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=candidate_version_number,
        key="validation_status",
        value="passed"
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=candidate_version_number,
        key="promotion_status",
        value="champion"
    )


    print(
        f"Version "
        f"{candidate_version_number} "
        f"is now CHAMPION."
    )


else:

    print("\n" + "=" * 80)
    print("PROMOTION FAILED")
    print("=" * 80)


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=candidate_version_number,
        key="validation_status",
        value="failed"
    )


    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=candidate_version_number,
        key="promotion_status",
        value="rejected"
    )


    print(
        "Candidate remains unchanged."
    )


# ============================================================
# 19. Final Registry verification
# ============================================================

if promotion_passed:

    champion_version = (
        client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS
        )
    )


    print("\n" + "=" * 80)
    print("CHAMPION VERIFICATION")
    print("=" * 80)


    print(
        f"Champion model   : "
        f"{champion_version.name}"
    )

    print(
        f"Champion version : "
        f"{champion_version.version}"
    )

    print(
        f"Champion status  : "
        f"{champion_version.status}"
    )


    champion_uri = (
        f"models:/"
        f"{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )


    print(
        f"Champion URI     : "
        f"{champion_uri}"
    )


print("\n" + "=" * 80)
print("VALIDATION WORKFLOW COMPLETED")
print("=" * 80)