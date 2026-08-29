import h5py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"


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
# 2. Selected 9-feature set
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
# 3. Load MATLAB data
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
# 4. Convert labels
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
# 5. Create DataFrame
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
# 6. X and y
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


print("\nSelected features:")

for feature in selected_features:
    print(" -", feature)


# ============================================================
# 7. ONE train-test split for BOTH models
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Test samples    : {len(X_test)}"
)


# ============================================================
# 8. Standardization for Logistic Regression ONLY
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 9. Logistic Regression
# ============================================================

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_pred = logistic_model.predict(
    X_test_scaled
)


# ============================================================
# 10. Random Forest
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_pred = random_forest_model.predict(
    X_test
)


# ============================================================
# 11. Evaluation function
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    y_pred
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
                y_pred
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred
            ),

        "F1":
            f1_score(
                y_true,
                y_pred
            )
    }


# ============================================================
# 12. Calculate metrics
# ============================================================

logistic_results = calculate_metrics(
    "Logistic Regression",
    y_test,
    logistic_pred
)


rf_results = calculate_metrics(
    "Random Forest",
    y_test,
    random_forest_pred
)


results = pd.DataFrame(
    [
        logistic_results,
        rf_results
    ]
)


# ============================================================
# 13. Model comparison
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 14. Logistic Regression report
# ============================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        logistic_pred,
        target_names=[
            "After (Normal)",
            "Before (Anomalous)"
        ]
    )
)


# ============================================================
# 15. Random Forest report
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        random_forest_pred,
        target_names=[
            "After (Normal)",
            "Before (Anomalous)"
        ]
    )
)


# ============================================================
# 16. Confusion matrices
# ============================================================

logistic_cm = confusion_matrix(
    y_test,
    logistic_pred
)


rf_cm = confusion_matrix(
    y_test,
    random_forest_pred
)


print("\n" + "=" * 80)
print("LOGISTIC REGRESSION CONFUSION MATRIX")
print("=" * 80)

print(logistic_cm)


print("\n" + "=" * 80)
print("RANDOM FOREST CONFUSION MATRIX")
print("=" * 80)

print(rf_cm)


# ============================================================
# 17. Logistic confusion matrix figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=[
        "After",
        "Before"
    ]
)


display.plot(
    ax=ax
)


ax.set_title(
    "Logistic Regression - Confusion Matrix"
)

fig.tight_layout()


# ============================================================
# 18. Random Forest confusion matrix figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=[
        "After",
        "Before"
    ]
)


display.plot(
    ax=ax
)


ax.set_title(
    "Random Forest - Confusion Matrix"
)

fig.tight_layout()


# ============================================================
# 19. Random Forest feature importance
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature":
            selected_features,

        "Importance":
            random_forest_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# 20. Feature importance plot
# ============================================================

plot_importance = (
    feature_importance
    .sort_values(
        by="Importance"
    )
)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.barh(
    plot_importance["Feature"],
    plot_importance["Importance"]
)


ax.set_xlabel(
    "Feature Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Random Forest Feature Importance"
)


fig.tight_layout()


# ============================================================
# 21. Show all figures
# ============================================================

plt.show()