import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

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
# 2. Original feature names
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
# 8. First split:
# 60% Train
# 40% Temporary
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=random_state,
    stratify=y
)


# ============================================================
# 9. Second split:
# Temporary 40% -> 20% Validation + 20% Test
# ============================================================

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


print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# 10. Use ONLY normal training samples
# ============================================================

normal_train_mask = (
    y_train == 0
)

X_train_normal = X_train.loc[
    normal_train_mask
]


print("\n" + "=" * 80)
print("ISOLATION FOREST TRAINING DATA")
print("=" * 80)

print(
    f"Normal training samples: "
    f"{len(X_train_normal)}"
)

print(
    "Anomalous training samples: 0"
)


# ============================================================
# 11. Build Isolation Forest
# ============================================================

model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=random_state,
    n_jobs=-1
)


# ============================================================
# 12. Train only on Normal data
# ============================================================

model.fit(
    X_train_normal
)


# ============================================================
# 13. Validation anomaly scores
# ============================================================

validation_scores = (
    -model.decision_function(
        X_val
    )
)


# ============================================================
# 14. Threshold candidates
# ============================================================

thresholds = np.linspace(
    validation_scores.min(),
    validation_scores.max(),
    500
)


# ============================================================
# 15. Search threshold maximizing F1
# ============================================================

threshold_results = []


for threshold in thresholds:

    y_val_pred = (
        validation_scores
        > threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    threshold_results.append(
        {
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 16. Best threshold based on F1
# ============================================================

best_row = (
    threshold_df
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


best_threshold = (
    best_row["Threshold"]
)


print("\n" + "=" * 80)
print("BEST VALIDATION THRESHOLD")
print("=" * 80)

print(
    f"Threshold : "
    f"{best_threshold:.6f}"
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


# ============================================================
# 17. Threshold with Recall >= 0.99
# ============================================================

high_recall_candidates = (
    threshold_df[
        threshold_df["Recall"] >= 0.99
    ]
)


if not high_recall_candidates.empty:

    high_recall_row = (
        high_recall_candidates
        .sort_values(
            by="Precision",
            ascending=False
        )
        .iloc[0]
    )

    high_recall_threshold = (
        high_recall_row["Threshold"]
    )

    print("\n" + "=" * 80)
    print("BEST THRESHOLD WITH RECALL >= 0.99")
    print("=" * 80)

    print(
        f"Threshold : "
        f"{high_recall_threshold:.6f}"
    )

    print(
        f"Precision : "
        f"{high_recall_row['Precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{high_recall_row['Recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{high_recall_row['F1']:.6f}"
    )

else:

    high_recall_threshold = None

    print(
        "\nNo threshold achieved "
        "Recall >= 0.99"
    )


# ============================================================
# 18. Final TEST anomaly scores
# ============================================================

test_scores = (
    -model.decision_function(
        X_test
    )
)


# ============================================================
# 19. Test predictions using BEST F1 threshold
# ============================================================

y_test_pred = (
    test_scores
    > best_threshold
).astype(int)


# ============================================================
# 20. Final TEST metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_scores
)

pr_auc = average_precision_score(
    y_test,
    test_scores
)


print("\n" + "=" * 80)
print("FINAL TEST PERFORMANCE")
print("=" * 80)

print(
    f"Accuracy : {accuracy:.6f}"
)

print(
    f"Precision: {precision:.6f}"
)

print(
    f"Recall   : {recall:.6f}"
)

print(
    f"F1       : {f1:.6f}"
)

print(
    f"ROC-AUC  : {roc_auc:.6f}"
)

print(
    f"PR-AUC   : {pr_auc:.6f}"
)


# ============================================================
# 21. Classification report
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "After (Normal)",
            "Before (Anomalous)"
        ],
        zero_division=0
    )
)


# ============================================================
# 22. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\n" + "=" * 80)
print("FINAL TEST CONFUSION MATRIX")
print("=" * 80)

print(cm)


# ============================================================
# 23. Confusion matrix plot
# ============================================================

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
    "Isolation Forest - Optimized Threshold"
)


fig.tight_layout()


# ============================================================
# 24. Validation Precision / Recall / F1 vs Threshold
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1"
)


ax.axvline(
    x=best_threshold,
    linestyle="--",
    linewidth=2,
    label="Best F1 Threshold"
)


ax.set_xlabel(
    "Anomaly Threshold"
)

ax.set_ylabel(
    "Metric Value"
)

ax.set_title(
    "Validation Metrics vs Anomaly Threshold"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 25. Test anomaly score distribution
# ============================================================

score_df = pd.DataFrame(
    {
        "True_Label": y_test.values,
        "Anomaly_Score": test_scores
    }
)


normal_scores = score_df.loc[
    score_df["True_Label"] == 0,
    "Anomaly_Score"
]


anomaly_scores = score_df.loc[
    score_df["True_Label"] == 1,
    "Anomaly_Score"
]


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    normal_scores,
    bins=50,
    alpha=0.6,
    label="After (Normal)"
)


ax.hist(
    anomaly_scores,
    bins=50,
    alpha=0.6,
    label="Before (Anomalous)"
)


ax.axvline(
    x=best_threshold,
    linestyle="--",
    linewidth=2,
    label="Optimized Threshold"
)


ax.set_xlabel(
    "Anomaly Score"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "Final Test - Anomaly Score Distribution"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 26. Show figures
# ============================================================

plt.show()