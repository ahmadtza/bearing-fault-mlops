import h5py
import pandas as pd

from sklearn.model_selection import train_test_split
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

file_path = "data/FeatureEntire.mat"


# All 12 original features
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


# MATLAB HDF5 objects corresponding to the 12 features
feature_objects = [
    "i", "j", "k", "l",
    "m", "n", "o", "p",
    "q", "r", "s", "t"
]


# Reduced feature set
# Removed:
# ch1_Std   because it is highly correlated with ch1_RMS
# ch2_Std   because it is highly correlated with ch2_RMS
# ch3_SINAD because it is highly correlated with ch3_SNR

reduced_features = [
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
# 2. Load MATLAB feature data
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    data = {}

    for feature_name, object_name in zip(
        feature_names,
        feature_objects
    ):

        values = refs[object_name][()].flatten()

        data[feature_name] = values


    # Read categorical label codes
    label_codes = refs["f"][()].flatten()


# ============================================================
# 3. Convert MATLAB labels
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
# 4. Create DataFrame
# ============================================================

df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


# ============================================================
# 5. Basic dataset information
# ============================================================

print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 6. Target encoding
# ============================================================

# 0 = After  = Normal
# 1 = Before = Anomalous

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
# 7. Prepare the two feature sets
# ============================================================

X_all = df[feature_names]

X_reduced = df[reduced_features]


print("\n" + "=" * 80)
print("FEATURE SETS")
print("=" * 80)

print("\nAll features:")
print(f"Number of features: {X_all.shape[1]}")

for feature in feature_names:
    print(" -", feature)


print("\nReduced features:")
print(f"Number of features: {X_reduced.shape[1]}")

for feature in reduced_features:
    print(" -", feature)


# ============================================================
# 8. Reusable Logistic Regression evaluation function
# ============================================================

def evaluate_logistic(X, y, model_name):

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "Model": model_name,
        "Number_of_Features": X.shape[1],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# 9. Evaluate model with all 12 features
# ============================================================

result_all = evaluate_logistic(
    X_all,
    y,
    "Logistic - All Features"
)


# ============================================================
# 10. Evaluate model with reduced 9 features
# ============================================================

result_reduced = evaluate_logistic(
    X_reduced,
    y,
    "Logistic - Reduced Features"
)


# ============================================================
# 11. Combine results
# ============================================================

results = pd.DataFrame(
    [
        result_all,
        result_reduced
    ]
)


# ============================================================
# 12. Display results
# ============================================================

print("\n" + "=" * 80)
print("FEATURE REDUNDANCY EXPERIMENT")
print("=" * 80)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 13. Compare performance differences
# ============================================================

accuracy_difference = (
    result_reduced["Accuracy"]
    - result_all["Accuracy"]
)

precision_difference = (
    result_reduced["Precision"]
    - result_all["Precision"]
)

recall_difference = (
    result_reduced["Recall"]
    - result_all["Recall"]
)

f1_difference = (
    result_reduced["F1"]
    - result_all["F1"]
)


print("\n" + "=" * 80)
print("PERFORMANCE DIFFERENCE")
print("=" * 80)

print(
    f"Accuracy difference : "
    f"{accuracy_difference:+.6f}"
)

print(
    f"Precision difference: "
    f"{precision_difference:+.6f}"
)

print(
    f"Recall difference   : "
    f"{recall_difference:+.6f}"
)

print(
    f"F1 difference       : "
    f"{f1_difference:+.6f}"
)


# ============================================================
# 14. Final summary
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Original model uses "
    f"{len(feature_names)} features."
)

print(
    f"Reduced model uses "
    f"{len(reduced_features)} features."
)

print(
    f"Removed features: "
    f"{len(feature_names) - len(reduced_features)}"
)

print(
    "\nRemoved redundant features:"
)

print(" - ch1_Std")
print(" - ch2_Std")
print(" - ch3_SINAD")