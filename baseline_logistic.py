import h5py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

feature_names = [
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
# 2. Load feature data
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    data = {}

    for feature_name, object_name in zip(
        feature_names,
        feature_objects
    ):
        data[feature_name] = (
            refs[object_name][()]
            .flatten()
        )

    label_codes = refs["f"][()].flatten()


# ============================================================
# 3. Convert labels
# ============================================================

label_map = {
    1: "Before",   # anomalous
    2: "After"     # normal
}

labels = [
    label_map[int(code)]
    for code in label_codes
]


# ============================================================
# 4. Create DataFrame
# ============================================================

df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


print("=" * 80)
print("DATASET")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 5. Define X and y
# ============================================================

X = df[feature_names]

# Binary encoding:
# 0 = After  = normal
# 1 = Before = anomalous
y = df["label"].map(
    {
        "After": 0,
        "Before": 1
    }
)


print("\n" + "=" * 80)
print("TARGET ENCODING")
print("=" * 80)

print("0 = After  = Normal")
print("1 = Before = Anomalous")


# ============================================================
# 6. Train-test split
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

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# 7. Standardization
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 8. Logistic Regression model
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 9. Predictions
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# 10. Evaluation metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 11. Classification report
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
        ]
    )
)


# ============================================================
# 12. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "After",
        "Before"
    ]
)

display.plot()

plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.tight_layout()

# ============================================================
# 13. Logistic Regression coefficients
# ============================================================

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)


print("\n" + "=" * 80)
print("LOGISTIC REGRESSION FEATURE COEFFICIENTS")
print("=" * 80)

print(
    coefficients.to_string(
        index=False
    )
)


# ============================================================
# 14. Plot feature coefficients
# ============================================================

plot_data = coefficients.sort_values(
    by="Coefficient"
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    plot_data["Feature"],
    plot_data["Coefficient"]
)

ax.axvline(
    x=0,
    linewidth=1
)

ax.set_xlabel(
    "Standardized Logistic Regression Coefficient"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Logistic Regression Feature Coefficients"
)

fig.tight_layout()
plt.show()