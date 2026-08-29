import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
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
    average_precision_score
)


# ============================================================
# 1. Configuration
# ============================================================

#file_path = "data/grouped_features.csv"
file_path = "data/grouped_time_frequency_hz_features.csv"
random_state = 42
n_splits = 5


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(file_path)


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


X = df[feature_columns]

y = df["label"]

groups = df["run_id"]


print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print(f"\nSamples : {len(df)}")
print(f"Runs    : {groups.nunique()}")
print(f"Features: {len(feature_columns)}")

print("\nRun-level class distribution:")

run_labels = (
    df.groupby("run_id")["label"]
    .first()
)

print(
    run_labels.value_counts()
)


# ============================================================
# 3. Stratified Group K-Fold
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)


# ============================================================
# 4. Results containers
# ============================================================

window_results = []

run_results = []


# ============================================================
# 5. Cross-validation loop
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print("\n" + "=" * 80)
    print(f"FOLD {fold}")
    print("=" * 80)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]


    train_runs = sorted(
        groups_train.unique()
    )

    test_runs = sorted(
        groups_test.unique()
    )


    print("\nTrain runs:")
    print(train_runs)

    print("\nTest runs:")
    print(test_runs)


    overlap = (
        set(train_runs)
        .intersection(test_runs)
    )

    assert len(overlap) == 0


    # --------------------------------------------------------
    # Scaling for Logistic and SVM
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # Logistic Regression
    # ========================================================

    logistic = LogisticRegression(
        max_iter=3000,
        random_state=random_state
    )

    logistic.fit(
        X_train_scaled,
        y_train
    )

    logistic_pred = logistic.predict(
        X_test_scaled
    )

    logistic_score = logistic.predict_proba(
        X_test_scaled
    )[:, 1]


    # ========================================================
    # Random Forest
    # ========================================================

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    rf_score = rf.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # SVM-RBF
    #
    # probability=False avoids the sklearn deprecation warning.
    # decision_function is enough for ROC-AUC / PR-AUC.
    # ========================================================

    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    svm.fit(
        X_train_scaled,
        y_train
    )

    svm_pred = svm.predict(
        X_test_scaled
    )

    svm_score = svm.decision_function(
        X_test_scaled
    )


    # ========================================================
    # Model dictionary
    # ========================================================

    models = {
        "Logistic Regression": (
            logistic_pred,
            logistic_score
        ),

        "Random Forest": (
            rf_pred,
            rf_score
        ),

        "SVM-RBF": (
            svm_pred,
            svm_score
        )
    }


    # ========================================================
    # 6. Window-level evaluation
    # ========================================================

    for model_name, (
        prediction,
        score
    ) in models.items():

        window_results.append(
            {
                "Fold": fold,
                "Model": model_name,

                "Accuracy":
                    accuracy_score(
                        y_test,
                        prediction
                    ),

                "Precision":
                    precision_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "Recall":
                    recall_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "F1":
                    f1_score(
                        y_test,
                        prediction,
                        zero_division=0
                    ),

                "ROC_AUC":
                    roc_auc_score(
                        y_test,
                        score
                    ),

                "PR_AUC":
                    average_precision_score(
                        y_test,
                        score
                    )
            }
        )


    # ========================================================
    # 7. Run-level evaluation
    #
    # Aggregate all windows belonging to one recording.
    # Mean anomaly/class score is used.
    # ========================================================

    test_metadata = pd.DataFrame(
        {
            "run_id":
                groups_test.values,

            "true_label":
                y_test.values
        }
    )


    for model_name, (
        prediction,
        score
    ) in models.items():

        run_df = test_metadata.copy()

        run_df["score"] = score


        # Average score across all 67 windows
        # belonging to each recording
        run_summary = (
            run_df
            .groupby("run_id")
            .agg(
                true_label=(
                    "true_label",
                    "first"
                ),

                mean_score=(
                    "score",
                    "mean"
                )
            )
            .reset_index()
        )


        # Logistic / RF probability threshold
        if model_name in [
            "Logistic Regression",
            "Random Forest"
        ]:

            run_summary["prediction"] = (
                run_summary["mean_score"]
                >= 0.5
            ).astype(int)

        # SVM decision-function boundary
        else:

            run_summary["prediction"] = (
                run_summary["mean_score"]
                >= 0
            ).astype(int)


        run_results.append(
            {
                "Fold": fold,
                "Model": model_name,

                "Accuracy":
                    accuracy_score(
                        run_summary["true_label"],
                        run_summary["prediction"]
                    ),

                "Precision":
                    precision_score(
                        run_summary["true_label"],
                        run_summary["prediction"],
                        zero_division=0
                    ),

                "Recall":
                    recall_score(
                        run_summary["true_label"],
                        run_summary["prediction"],
                        zero_division=0
                    ),

                "F1":
                    f1_score(
                        run_summary["true_label"],
                        run_summary["prediction"],
                        zero_division=0
                    ),

                "ROC_AUC":
                    roc_auc_score(
                        run_summary["true_label"],
                        run_summary["mean_score"]
                    ),

                "PR_AUC":
                    average_precision_score(
                        run_summary["true_label"],
                        run_summary["mean_score"]
                    )
            }
        )


# ============================================================
# 8. Convert results to DataFrames
# ============================================================

window_results_df = pd.DataFrame(
    window_results
)

run_results_df = pd.DataFrame(
    run_results
)


# ============================================================
# 9. Window-level fold results
# ============================================================

print("\n" + "=" * 100)
print("WINDOW-LEVEL RESULTS FOR ALL FOLDS")
print("=" * 100)

print(
    window_results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. Window-level summary
# ============================================================

window_summary = (
    window_results_df
    .groupby("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ]
    ]
    .agg(
        ["mean", "std"]
    )
)


print("\n" + "=" * 100)
print("WINDOW-LEVEL CROSS-VALIDATION SUMMARY")
print("=" * 100)

print(window_summary)


# ============================================================
# 11. Run-level results
# ============================================================

print("\n" + "=" * 100)
print("RUN-LEVEL RESULTS FOR ALL FOLDS")
print("=" * 100)

print(
    run_results_df.to_string(
        index=False
    )
)


# ============================================================
# 12. Run-level summary
# ============================================================

run_summary = (
    run_results_df
    .groupby("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ]
    ]
    .agg(
        ["mean", "std"]
    )
)


print("\n" + "=" * 100)
print("RUN-LEVEL CROSS-VALIDATION SUMMARY")
print("=" * 100)

print(run_summary)


# ============================================================
# 13. Best model based on mean run-level F1
# ============================================================

mean_run_results = (
    run_results_df
    .groupby("Model")
    .mean(
        numeric_only=True
    )
)


best_model_name = (
    mean_run_results["F1"]
    .idxmax()
)


best_model = (
    mean_run_results
    .loc[best_model_name]
)


print("\n" + "=" * 80)
print("BEST MODEL BASED ON RUN-LEVEL F1")
print("=" * 80)

print(
    f"Model        : {best_model_name}"
)

print(
    f"Mean Accuracy: "
    f"{best_model['Accuracy']:.6f}"
)

print(
    f"Mean Recall  : "
    f"{best_model['Recall']:.6f}"
)

print(
    f"Mean F1      : "
    f"{best_model['F1']:.6f}"
)

print(
    f"Mean ROC-AUC : "
    f"{best_model['ROC_AUC']:.6f}"
)

print(
    f"Mean PR-AUC  : "
    f"{best_model['PR_AUC']:.6f}"
)