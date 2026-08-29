import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"

random_state = 42


# ============================================================
# 2. All features
# ============================================================

all_features = [
    "ch1_CrestFactor",
    "ch1_Kurtosis",
    "ch1_RMS",
    "ch1_Std",
    "ch2_Mean",
    "ch2_RMS",
    "ch2_Skewness",
    "ch2_Std",
    "ch3_CrestFactor",
    "ch3_SINAD",
    "ch3_SNR",
    "ch3_THD"
]


feature_objects = [
    "i", "j", "k", "l",
    "m", "n", "o", "p",
    "q", "r", "s", "t"
]


# ============================================================
# 3. Selected 9 features
# ============================================================

selected_features = [
    "ch1_CrestFactor",
    "ch1_Kurtosis",
    "ch1_RMS",

    "ch2_Mean",
    "ch2_RMS",
    "ch2_Skewness",

    "ch3_CrestFactor",
    "ch3_SNR",
    "ch3_THD"
]


# ============================================================
# 4. Load MATLAB data
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    data = {}

    for feature_name, object_name in zip(
        all_features,
        feature_objects
    ):
        data[feature_name] = (
            refs[object_name][()]
            .flatten()
        )

    label_codes = (
        refs["f"][()]
        .flatten()
    )


# ============================================================
# 5. Convert labels
# ============================================================

label_map = {
    1: "Before",
    2: "After"
}


labels = [
    label_map[int(code)]
    for code in label_codes
]


# ============================================================
# 6. Create DataFrame
# ============================================================

df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


# ============================================================
# 7. Define X and y
# ============================================================

X = df[selected_features]

# 0 = After  = Normal
# 1 = Before = Anomalous

y = df["label"].map(
    {
        "After": 0,
        "Before": 1
    }
)


print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 8. Train / Validation / Test split
#
# 60% Train
# 20% Validation
# 20% Test
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=random_state,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=random_state,
    stratify=y_temp
)


print("\n" + "=" * 80)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

print(
    f"Train samples     : {len(X_train)}"
)

print(
    f"Validation samples: {len(X_val)}"
)

print(
    f"Test samples      : {len(X_test)}"
)


# ============================================================
# 9. Keep ONLY normal samples for training
# ============================================================

normal_mask = (
    y_train == 0
)

X_train_normal = X_train.loc[
    normal_mask
]


print("\n" + "=" * 80)
print("NORMAL-ONLY TRAINING")
print("=" * 80)

print(
    f"Normal training samples: "
    f"{len(X_train_normal)}"
)

print(
    "Anomalous training samples: 0"
)


# ============================================================
# 10. Scaling for One-Class SVM
# ============================================================

scaler = StandardScaler()

X_train_normal_scaled = scaler.fit_transform(
    X_train_normal
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. Isolation Forest
# ============================================================

isolation_model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=random_state,
    n_jobs=-1
)


isolation_model.fit(
    X_train_normal
)


# ============================================================
# 12. One-Class SVM
# ============================================================

ocsvm_model = OneClassSVM(
    kernel="rbf",
    nu=0.05,
    gamma="scale"
)


ocsvm_model.fit(
    X_train_normal_scaled
)


# ============================================================
# 13. Validation anomaly scores
#
# Larger score = more anomalous
# ============================================================

isolation_val_scores = (
    -isolation_model.decision_function(
        X_val
    )
)


ocsvm_val_scores = (
    -ocsvm_model.decision_function(
        X_val_scaled
    )
)


# ============================================================
# 14. Threshold optimization function
# ============================================================

def optimize_threshold(
    y_true,
    anomaly_scores,
    number_of_thresholds=500
):

    thresholds = np.linspace(
        anomaly_scores.min(),
        anomaly_scores.max(),
        number_of_thresholds
    )

    rows = []

    for threshold in thresholds:

        y_pred = (
            anomaly_scores
            > threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
        )

    results = pd.DataFrame(
        rows
    )

    best_row = (
        results
        .sort_values(
            by="F1",
            ascending=False
        )
        .iloc[0]
    )

    return (
        best_row["Threshold"],
        best_row,
        results
    )


# ============================================================
# 15. Optimize Isolation Forest threshold
# ============================================================

(
    isolation_threshold,
    isolation_best_validation,
    isolation_threshold_df
) = optimize_threshold(
    y_val,
    isolation_val_scores
)


# ============================================================
# 16. Optimize One-Class SVM threshold
# ============================================================

(
    ocsvm_threshold,
    ocsvm_best_validation,
    ocsvm_threshold_df
) = optimize_threshold(
    y_val,
    ocsvm_val_scores
)


# ============================================================
# 17. Print validation thresholds
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION THRESHOLD OPTIMIZATION")
print("=" * 80)


print("\nIsolation Forest:")

print(
    f"Threshold : "
    f"{isolation_threshold:.6f}"
)

print(
    f"Precision : "
    f"{isolation_best_validation['Precision']:.6f}"
)

print(
    f"Recall    : "
    f"{isolation_best_validation['Recall']:.6f}"
)

print(
    f"F1        : "
    f"{isolation_best_validation['F1']:.6f}"
)


print("\nOne-Class SVM:")

print(
    f"Threshold : "
    f"{ocsvm_threshold:.6f}"
)

print(
    f"Precision : "
    f"{ocsvm_best_validation['Precision']:.6f}"
)

print(
    f"Recall    : "
    f"{ocsvm_best_validation['Recall']:.6f}"
)

print(
    f"F1        : "
    f"{ocsvm_best_validation['F1']:.6f}"
)


# ============================================================
# 18. Final TEST anomaly scores
# ============================================================

isolation_test_scores = (
    -isolation_model.decision_function(
        X_test
    )
)


ocsvm_test_scores = (
    -ocsvm_model.decision_function(
        X_test_scaled
    )
)


# ============================================================
# 19. Final TEST predictions
# ============================================================

isolation_test_pred = (
    isolation_test_scores
    > isolation_threshold
).astype(int)


ocsvm_test_pred = (
    ocsvm_test_scores
    > ocsvm_threshold
).astype(int)


# ============================================================
# 20. Evaluation function
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    scores
):

    return {
        "Model": model_name,

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                scores
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                scores
            )
    }


# ============================================================
# 21. Evaluate both models
# ============================================================

isolation_results = evaluate_model(
    "Isolation Forest",
    y_test,
    isolation_test_pred,
    isolation_test_scores
)


ocsvm_results = evaluate_model(
    "One-Class SVM",
    y_test,
    ocsvm_test_pred,
    ocsvm_test_scores
)


results = pd.DataFrame(
    [
        isolation_results,
        ocsvm_results
    ]
)


# ============================================================
# 22. Final model comparison
# ============================================================

print("\n" + "=" * 100)
print("ANOMALY MODEL COMPARISON - FINAL TEST")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 23. Classification reports
# ============================================================

models_predictions = {
    "Isolation Forest":
        isolation_test_pred,

    "One-Class SVM":
        ocsvm_test_pred
}


for model_name, prediction in models_predictions.items():

    print("\n" + "=" * 80)
    print(model_name.upper())
    print("=" * 80)

    print(
        classification_report(
            y_test,
            prediction,
            target_names=[
                "After (Normal)",
                "Before (Anomalous)"
            ],
            zero_division=0
        )
    )


# ============================================================
# 24. Confusion matrices
# ============================================================

for model_name, prediction in models_predictions.items():

    cm = confusion_matrix(
        y_test,
        prediction
    )

    print("\n" + "=" * 80)
    print(
        f"{model_name.upper()} CONFUSION MATRIX"
    )
    print("=" * 80)

    print(cm)

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "After",
            "Before"
        ]
    )

    display.plot(
        ax=ax
    )

    ax.set_title(
        f"{model_name} - Final Test"
    )

    fig.tight_layout()


# ============================================================
# 25. Threshold curves - Isolation Forest
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.plot(
    isolation_threshold_df["Threshold"],
    isolation_threshold_df["Precision"],
    label="Precision"
)

ax.plot(
    isolation_threshold_df["Threshold"],
    isolation_threshold_df["Recall"],
    label="Recall"
)

ax.plot(
    isolation_threshold_df["Threshold"],
    isolation_threshold_df["F1"],
    label="F1"
)

ax.axvline(
    x=isolation_threshold,
    linestyle="--",
    linewidth=2,
    label="Best Threshold"
)

ax.set_xlabel(
    "Threshold"
)

ax.set_ylabel(
    "Metric"
)

ax.set_title(
    "Isolation Forest - Validation Threshold"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 26. Threshold curves - One-Class SVM
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.plot(
    ocsvm_threshold_df["Threshold"],
    ocsvm_threshold_df["Precision"],
    label="Precision"
)

ax.plot(
    ocsvm_threshold_df["Threshold"],
    ocsvm_threshold_df["Recall"],
    label="Recall"
)

ax.plot(
    ocsvm_threshold_df["Threshold"],
    ocsvm_threshold_df["F1"],
    label="F1"
)

ax.axvline(
    x=ocsvm_threshold,
    linestyle="--",
    linewidth=2,
    label="Best Threshold"
)

ax.set_xlabel(
    "Threshold"
)

ax.set_ylabel(
    "Metric"
)

ax.set_title(
    "One-Class SVM - Validation Threshold"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 27. Score distribution - Isolation Forest
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    isolation_test_scores[
        y_test.values == 0
    ],
    bins=50,
    alpha=0.6,
    label="After (Normal)"
)

ax.hist(
    isolation_test_scores[
        y_test.values == 1
    ],
    bins=50,
    alpha=0.6,
    label="Before (Anomalous)"
)

ax.axvline(
    x=isolation_threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

ax.set_xlabel(
    "Anomaly Score"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "Isolation Forest - Test Scores"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 28. Score distribution - One-Class SVM
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.hist(
    ocsvm_test_scores[
        y_test.values == 0
    ],
    bins=50,
    alpha=0.6,
    label="After (Normal)"
)

ax.hist(
    ocsvm_test_scores[
        y_test.values == 1
    ],
    bins=50,
    alpha=0.6,
    label="Before (Anomalous)"
)

ax.axvline(
    x=ocsvm_threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

ax.set_xlabel(
    "Anomaly Score"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "One-Class SVM - Test Scores"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 29. Best anomaly model
# ============================================================

best_row = (
    results
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST ANOMALY MODEL BASED ON F1")
print("=" * 80)

print(
    f"Model     : "
    f"{best_row['Model']}"
)

print(
    f"Accuracy  : "
    f"{best_row['Accuracy']:.6f}"
)

print(
    f"Precision : "
    f"{best_row['Precision']:.6f}"
)

print(
    f"Recall    : "
    f"{best_row['Recall']:.6f}"
)

print(
    f"F1        : "
    f"{best_row['F1']:.6f}"
)

print(
    f"ROC-AUC   : "
    f"{best_row['ROC_AUC']:.6f}"
)

print(
    f"PR-AUC    : "
    f"{best_row['PR_AUC']:.6f}"
)


# ============================================================
# 30. Show figures
# ============================================================

plt.show()