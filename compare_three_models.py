import h5py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
# 2. Selected 9 features
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


# ============================================================
# 6. Define X and y
# ============================================================

X = df[selected_features]


# Target:
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

print("\nSelected features:")

for feature in selected_features:
    print(" -", feature)


# ============================================================
# 7. One common Train/Test Split
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
    f"Training samples: {X_train.shape[0]}"
)

print(
    f"Test samples    : {X_test.shape[0]}"
)


# ============================================================
# 8. Scaling
# Logistic Regression and SVM need scaled data
# Random Forest does not
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

logistic_prob = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


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

rf_pred = random_forest_model.predict(
    X_test
)

rf_prob = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. SVM with RBF kernel
# ============================================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm_model.predict(
    X_test_scaled
)

svm_prob = svm_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 12. Evaluation function
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_score
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
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                y_score
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                y_score
            )
    }


# ============================================================
# 13. Calculate model metrics
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)


rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


svm_results = evaluate_model(
    "SVM - RBF",
    y_test,
    svm_pred,
    svm_prob
)


results = pd.DataFrame(
    [
        logistic_results,
        rf_results,
        svm_results
    ]
)


# ============================================================
# 14. Main comparison
# ============================================================

print("\n" + "=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 15. Detailed reports
# ============================================================

models_predictions = {
    "LOGISTIC REGRESSION":
        logistic_pred,

    "RANDOM FOREST":
        rf_pred,

    "SVM - RBF":
        svm_pred
}


for model_name, prediction in models_predictions.items():

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    print(
        classification_report(
            y_test,
            prediction,
            target_names=[
                "After (Normal)",
                "Before (Anomalous)"
            ]
        )
    )


# ============================================================
# 16. Confusion matrices
# ============================================================

for model_name, prediction in models_predictions.items():

    cm = confusion_matrix(
        y_test,
        prediction
    )

    print("\n" + "=" * 80)
    print(
        f"{model_name} CONFUSION MATRIX"
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
        f"{model_name} - Confusion Matrix"
    )

    fig.tight_layout()


# ============================================================
# 17. Random Forest Feature Importance
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
# 18. Random Forest feature importance plot
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
# 19. Logistic Regression coefficients
# ============================================================

logistic_coefficients = pd.DataFrame(
    {
        "Feature":
            selected_features,

        "Coefficient":
            logistic_model.coef_[0]
    }
)


logistic_coefficients[
    "Absolute_Coefficient"
] = (
    logistic_coefficients[
        "Coefficient"
    ].abs()
)


logistic_coefficients = (
    logistic_coefficients
    .sort_values(
        by="Absolute_Coefficient",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 80)

print(
    logistic_coefficients.to_string(
        index=False
    )
)


# ============================================================
# 20. Best model
# ============================================================

best_model_row = (
    results
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST MODEL BASED ON F1")
print("=" * 80)

print(
    f"Model     : "
    f"{best_model_row['Model']}"
)

print(
    f"Accuracy  : "
    f"{best_model_row['Accuracy']:.6f}"
)

print(
    f"Precision : "
    f"{best_model_row['Precision']:.6f}"
)

print(
    f"Recall    : "
    f"{best_model_row['Recall']:.6f}"
)

print(
    f"F1        : "
    f"{best_model_row['F1']:.6f}"
)

print(
    f"ROC-AUC   : "
    f"{best_model_row['ROC_AUC']:.6f}"
)

print(
    f"PR-AUC    : "
    f"{best_model_row['PR_AUC']:.6f}"
)


# ============================================================
# 21. Show figures
# ============================================================

plt.show()