import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"

random_state = 42


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
# 2. Load data
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

    label_codes = refs["f"][()].flatten()


label_map = {
    1: "Before",
    2: "After"
}


labels = [
    label_map[int(code)]
    for code in label_codes
]


df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


# ============================================================
# 3. X and y
# ============================================================

X = df[selected_features]

# 0 = Normal
# 1 = Anomalous

y = df["label"].map(
    {
        "After": 0,
        "Before": 1
    }
)


# ============================================================
# 4. Train / Validation / Test
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


# ============================================================
# 5. Normal-only training
# ============================================================

X_train_normal = X_train.loc[
    y_train == 0
]


print("=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(
    "Normal training samples:",
    len(X_train_normal)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Final test samples:",
    len(X_test)
)


# ============================================================
# 6. Standardization
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
# 7. Hyperparameter grid
# ============================================================

nu_values = [
    0.01,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15
]


gamma_values = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00
]


# ============================================================
# 8. Threshold optimizer
# ============================================================

def find_best_threshold(
    y_true,
    scores
):

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        500
    )

    best_threshold = None
    best_f1 = -1
    best_precision = None
    best_recall = None

    for threshold in thresholds:

        prediction = (
            scores > threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

            best_precision = precision_score(
                y_true,
                prediction,
                zero_division=0
            )

            best_recall = recall_score(
                y_true,
                prediction,
                zero_division=0
            )

    return (
        best_threshold,
        best_precision,
        best_recall,
        best_f1
    )


# ============================================================
# 9. Hyperparameter search
# ============================================================

search_results = []


print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH")
print("=" * 80)


for nu in nu_values:

    for gamma in gamma_values:

        model = OneClassSVM(
            kernel="rbf",
            nu=nu,
            gamma=gamma
        )


        model.fit(
            X_train_normal_scaled
        )


        val_scores = (
            -model.decision_function(
                X_val_scaled
            )
        )


        (
            threshold,
            precision,
            recall,
            f1
        ) = find_best_threshold(
            y_val,
            val_scores
        )


        search_results.append(
            {
                "nu": nu,
                "gamma": gamma,
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
        )


# ============================================================
# 10. Results table
# ============================================================

search_df = pd.DataFrame(
    search_results
)


search_df = search_df.sort_values(
    by="F1",
    ascending=False
)


print("\nTop 10 configurations:")

print(
    search_df
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. Best configuration
# ============================================================

best = search_df.iloc[0]


best_nu = best["nu"]
best_gamma = best["gamma"]
best_threshold = best["Threshold"]


print("\n" + "=" * 80)
print("BEST VALIDATION CONFIGURATION")
print("=" * 80)

print(
    f"nu        : {best_nu}"
)

print(
    f"gamma     : {best_gamma}"
)

print(
    f"Threshold : {best_threshold:.6f}"
)

print(
    f"Precision : {best['Precision']:.6f}"
)

print(
    f"Recall    : {best['Recall']:.6f}"
)

print(
    f"F1        : {best['F1']:.6f}"
)


# ============================================================
# 12. Train final selected One-Class SVM
# ============================================================

best_model = OneClassSVM(
    kernel="rbf",
    nu=best_nu,
    gamma=best_gamma
)


best_model.fit(
    X_train_normal_scaled
)


# ============================================================
# 13. Final TEST scores
# ============================================================

test_scores = (
    -best_model.decision_function(
        X_test_scaled
    )
)


# ============================================================
# 14. Final TEST predictions
# ============================================================

y_test_pred = (
    test_scores > best_threshold
).astype(int)


# ============================================================
# 15. Final metrics
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
# 16. Classification report
# ============================================================

print("\n" + "=" * 80)
print("FINAL CLASSIFICATION REPORT")
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
# 17. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\n" + "=" * 80)
print("FINAL CONFUSION MATRIX")
print("=" * 80)

print(cm)