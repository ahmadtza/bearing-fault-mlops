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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"

random_state = 42

test_size = 0.20


# ============================================================
# 2. All features stored in MATLAB file
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


print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 7. Define X and y
# ============================================================

X = df[selected_features]

# Our evaluation convention:
#
# 0 = After  = Normal
# 1 = Before = Anomalous

y = df["label"].map(
    {
        "After": 0,
        "Before": 1
    }
)


# ============================================================
# 8. Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)


print("\n" + "=" * 80)
print("ORIGINAL TRAIN / TEST SPLIT")
print("=" * 80)

print(
    f"Original training samples: "
    f"{len(X_train)}"
)

print(
    f"Test samples             : "
    f"{len(X_test)}"
)

print("\nOriginal training class distribution:")

print(
    y_train.value_counts()
)


# ============================================================
# 9. Keep ONLY normal samples for training
# ============================================================

normal_train_mask = (
    y_train == 0
)

X_train_normal = X_train.loc[
    normal_train_mask
]


print("\n" + "=" * 80)
print("ANOMALY DETECTION TRAINING DATA")
print("=" * 80)

print(
    "Only After (Normal) samples "
    "are used for training."
)

print(
    f"\nNormal training samples: "
    f"{len(X_train_normal)}"
)

print(
    f"Anomalous samples used "
    f"during training: 0"
)


# ============================================================
# 10. Build Isolation Forest
# ============================================================

model = IsolationForest(
    n_estimators=300,

    contamination="auto",

    random_state=random_state,

    n_jobs=-1
)


# ============================================================
# 11. Train ONLY on normal data
# ============================================================

model.fit(
    X_train_normal
)


# ============================================================
# 12. Prediction on mixed Test set
# ============================================================

# IsolationForest returns:
#
# +1 = inlier
# -1 = outlier

raw_prediction = model.predict(
    X_test
)


# Convert to our convention:
#
# 0 = Normal
# 1 = Anomalous

y_pred = np.where(
    raw_prediction == -1,
    1,
    0
)


# ============================================================
# 13. Anomaly score
# ============================================================

# decision_function:
#
# positive → more normal
# negative → more anomalous
#
# We multiply by -1 so that:
#
# larger score → more anomalous

anomaly_score = -model.decision_function(
    X_test
)


# ============================================================
# 14. Evaluation metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    anomaly_score
)

pr_auc = average_precision_score(
    y_test,
    anomaly_score
)


# ============================================================
# 15. Print performance
# ============================================================

print("\n" + "=" * 80)
print("ISOLATION FOREST PERFORMANCE")
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
# 16. Classification report
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "After (Normal)",
            "Before (Anomalous)"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


# ============================================================
# 18. Confusion Matrix Plot
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
    "Isolation Forest - Confusion Matrix"
)

fig.tight_layout()


# ============================================================
# 19. Anomaly score statistics
# ============================================================

score_df = pd.DataFrame(
    {
        "True_Label": y_test.values,
        "Anomaly_Score": anomaly_score
    }
)


score_df["Condition"] = (
    score_df["True_Label"]
    .map(
        {
            0: "After (Normal)",
            1: "Before (Anomalous)"
        }
    )
)


print("\n" + "=" * 80)
print("ANOMALY SCORE STATISTICS")
print("=" * 80)

print(
    score_df
    .groupby("Condition")["Anomaly_Score"]
    .describe()
)


# ============================================================
# 20. Plot anomaly score distribution
# ============================================================

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


# Isolation Forest default boundary corresponds
# approximately to anomaly_score = 0

ax.axvline(
    x=0,
    linestyle="--",
    linewidth=2,
    label="Default Threshold"
)


ax.set_xlabel(
    "Anomaly Score"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "Isolation Forest - Anomaly Score Distribution"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 21. Show figures
# ============================================================

plt.show()