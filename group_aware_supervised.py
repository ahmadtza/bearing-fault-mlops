import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
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
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/grouped_features.csv"

random_state = 42


# ============================================================
# 2. Load grouped feature dataset
# ============================================================

df = pd.read_csv(file_path)


print("=" * 80)
print("DATASET")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nRuns:")
print(df["run_id"].nunique())

print("\nClass distribution:")
print(df["condition"].value_counts())


# ============================================================
# 3. Define feature columns
# ============================================================

metadata_columns = [
    "run_id",
    "window_id",
    "start_sample",
    "end_sample",
    "label",
    "condition"
]


feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print("\nNumber of features:")
print(len(feature_columns))


# ============================================================
# 4. X, y and groups
# ============================================================

X = df[feature_columns]

y = df["label"]

groups = df["run_id"]


# ============================================================
# 5. Group-aware Train/Test split
#
# 80% runs -> Train
# 20% runs -> Test
# ============================================================

group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=random_state
)


train_indices, test_indices = next(
    group_split.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[
    train_indices
]

X_test = X.iloc[
    test_indices
]

y_train = y.iloc[
    train_indices
]

y_test = y.iloc[
    test_indices
]

groups_train = groups.iloc[
    train_indices
]

groups_test = groups.iloc[
    test_indices
]


# ============================================================
# 6. Verify run separation
# ============================================================

train_runs = sorted(
    groups_train.unique()
)

test_runs = sorted(
    groups_test.unique()
)


print("\n" + "=" * 80)
print("GROUP SPLIT")
print("=" * 80)

print("\nTrain runs:")
print(train_runs)

print("\nTest runs:")
print(test_runs)

print("\nNumber of train runs:")
print(len(train_runs))

print("\nNumber of test runs:")
print(len(test_runs))


overlap = set(
    train_runs
).intersection(
    test_runs
)


print("\nRun overlap:")
print(overlap)


assert len(overlap) == 0


# ============================================================
# 7. Class distribution by RUN
# ============================================================

train_run_labels = (
    df[
        df["run_id"].isin(
            train_runs
        )
    ]
    .groupby("run_id")["label"]
    .first()
)


test_run_labels = (
    df[
        df["run_id"].isin(
            test_runs
        )
    ]
    .groupby("run_id")["label"]
    .first()
)


print("\nTrain run class distribution:")
print(
    train_run_labels.value_counts()
)


print("\nTest run class distribution:")
print(
    test_run_labels.value_counts()
)


# ============================================================
# 8. Standardization
#
# Logistic Regression and SVM only
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
    max_iter=3000,
    random_state=random_state
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_pred = logistic_model.predict(
    X_test_scaled
)


logistic_score = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 10. Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=random_state,
    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)


rf_pred = rf_model.predict(
    X_test
)


rf_score = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. SVM-RBF
# ============================================================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=random_state
)


svm_model.fit(
    X_train_scaled,
    y_train
)


svm_pred = svm_model.predict(
    X_test_scaled
)


svm_score = svm_model.predict_proba(
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
                y_score
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                y_score
            )
    }


# ============================================================
# 13. Collect results
# ============================================================

results = pd.DataFrame(
    [
        evaluate_model(
            "Logistic Regression",
            y_test,
            logistic_pred,
            logistic_score
        ),

        evaluate_model(
            "Random Forest",
            y_test,
            rf_pred,
            rf_score
        ),

        evaluate_model(
            "SVM-RBF",
            y_test,
            svm_pred,
            svm_score
        )
    ]
)


# ============================================================
# 14. Print comparison
# ============================================================

print("\n" + "=" * 100)
print("GROUP-AWARE MODEL COMPARISON")
print("=" * 100)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 15. Confusion matrices
# ============================================================

predictions = {
    "Logistic Regression":
        logistic_pred,

    "Random Forest":
        rf_pred,

    "SVM-RBF":
        svm_pred
}


for model_name, prediction in predictions.items():

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


# ============================================================
# 16. Detailed reports
# ============================================================

for model_name, prediction in predictions.items():

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
# 17. Random Forest feature importance
# ============================================================

feature_importance = pd.DataFrame(
    {
        "Feature":
            feature_columns,

        "Importance":
            rf_model.feature_importances_
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
    feature_importance
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# 18. Best model
# ============================================================

best_model = (
    results
    .sort_values(
        by="F1",
        ascending=False
    )
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST GROUP-AWARE MODEL")
print("=" * 80)

print(
    f"Model    : "
    f"{best_model['Model']}"
)

print(
    f"Accuracy : "
    f"{best_model['Accuracy']:.6f}"
)

print(
    f"F1       : "
    f"{best_model['F1']:.6f}"
)

print(
    f"ROC-AUC  : "
    f"{best_model['ROC_AUC']:.6f}"
)

print(
    f"PR-AUC   : "
    f"{best_model['PR_AUC']:.6f}"
)