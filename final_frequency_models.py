import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/grouped_time_frequency_hz_features.csv"

random_state = 42
n_splits = 5


# ============================================================
# 2. Final engineering feature set
# ============================================================

selected_features = [
    "ch3_SpectralCentroidHz",
    "ch3_BandEnergy_1000_2000",
    "ch3_RelEnergy_2000_4000",
    "ch3_SpectralFlatness",
    "ch3_BandEnergy_100_500",
    "ch3_RelEnergy_100_500",
    "ch3_BandEnergy_500_1000",
    "ch3_BandEnergy_2000_4000",
    "ch1_SpectralFlatness",
    "ch3_SpectralSpreadHz"
]


# ============================================================
# 3. Load dataset
# ============================================================

df = pd.read_csv(file_path)


print("=" * 90)
print("FINAL FREQUENCY MODEL PIPELINE")
print("=" * 90)


print("\nDataset shape:")
print(df.shape)


print("\nSelected features:")

for feature in selected_features:
    print(" -", feature)


# ============================================================
# 4. Define X, y and groups
# ============================================================

X = df[selected_features]

y = df["label"]

groups = df["run_id"]


print("\nNumber of windows:")
print(len(df))


print("\nNumber of runs:")
print(groups.nunique())


print("\nRun-level class distribution:")

run_label_summary = (
    df.groupby("run_id")["label"]
    .first()
)

print(
    run_label_summary.value_counts()
)


# ============================================================
# 5. Cross-validation strategy
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)


splits = list(
    cv.split(
        X,
        y,
        groups
    )
)


# ============================================================
# 6. Result containers
# ============================================================

window_results = []

run_results = []

all_window_predictions = []

all_run_predictions = []


# ============================================================
# 7. Cross-validation loop
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    splits,
    start=1
):

    print("\n" + "=" * 90)
    print(f"FOLD {fold}")
    print("=" * 90)


    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    groups_train = groups.iloc[
        train_idx
    ]

    groups_test = groups.iloc[
        test_idx
    ]


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
        .intersection(
            test_runs
        )
    )

    assert len(overlap) == 0


    # ========================================================
    # 8. Scaling
    # ========================================================

    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(
        X_train
    )


    X_test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # 9. Logistic Regression
    # ========================================================

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


    logistic_score = (
        logistic_model
        .predict_proba(
            X_test_scaled
        )[:, 1]
    )


    # ========================================================
    # 10. SVM-RBF
    # ========================================================

    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )


    svm_model.fit(
        X_train_scaled,
        y_train
    )


    svm_pred = svm_model.predict(
        X_test_scaled
    )


    svm_score = svm_model.decision_function(
        X_test_scaled
    )


    # ========================================================
    # 11. Random Forest
    # ========================================================

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


    rf_score = (
        rf_model
        .predict_proba(
            X_test
        )[:, 1]
    )


    # ========================================================
    # 12. Model dictionary
    # ========================================================

    models = {

        "Logistic Regression": (
            logistic_pred,
            logistic_score,
            0.5
        ),

        "SVM-RBF": (
            svm_pred,
            svm_score,
            0.0
        ),

        "Random Forest": (
            rf_pred,
            rf_score,
            0.5
        )
    }


    # ========================================================
    # 13. Evaluation loop
    # ========================================================

    for model_name, (
        prediction,
        score,
        run_threshold
    ) in models.items():

        # ----------------------------------------------------
        # Window-level metrics
        # ----------------------------------------------------

        window_accuracy = accuracy_score(
            y_test,
            prediction
        )

        window_precision = precision_score(
            y_test,
            prediction,
            zero_division=0
        )

        window_recall = recall_score(
            y_test,
            prediction,
            zero_division=0
        )

        window_f1 = f1_score(
            y_test,
            prediction,
            zero_division=0
        )

        window_roc_auc = roc_auc_score(
            y_test,
            score
        )

        window_pr_auc = average_precision_score(
            y_test,
            score
        )


        window_results.append(
            {
                "Fold": fold,
                "Model": model_name,
                "Accuracy": window_accuracy,
                "Precision": window_precision,
                "Recall": window_recall,
                "F1": window_f1,
                "ROC_AUC": window_roc_auc,
                "PR_AUC": window_pr_auc
            }
        )


        # ----------------------------------------------------
        # Save individual window predictions
        # ----------------------------------------------------

        fold_window_predictions = pd.DataFrame(
            {
                "fold": fold,

                "model":
                    model_name,

                "run_id":
                    groups_test.values,

                "true_label":
                    y_test.values,

                "prediction":
                    prediction,

                "score":
                    score
            }
        )


        all_window_predictions.append(
            fold_window_predictions
        )


        # ====================================================
        # 14. Run-level aggregation
        # ====================================================

        run_df = pd.DataFrame(
            {
                "run_id":
                    groups_test.values,

                "true_label":
                    y_test.values,

                "score":
                    score
            }
        )


        run_summary = (
            run_df
            .groupby(
                "run_id"
            )
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


        run_summary[
            "prediction"
        ] = (
            run_summary[
                "mean_score"
            ]
            >= run_threshold
        ).astype(int)


        # ----------------------------------------------------
        # Run-level metrics
        # ----------------------------------------------------

        run_accuracy = accuracy_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "prediction"
            ]
        )


        run_precision = precision_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "prediction"
            ],
            zero_division=0
        )


        run_recall = recall_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "prediction"
            ],
            zero_division=0
        )


        run_f1 = f1_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "prediction"
            ],
            zero_division=0
        )


        run_roc_auc = roc_auc_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "mean_score"
            ]
        )


        run_pr_auc = average_precision_score(
            run_summary[
                "true_label"
            ],
            run_summary[
                "mean_score"
            ]
        )


        run_results.append(
            {
                "Fold": fold,
                "Model": model_name,
                "Accuracy": run_accuracy,
                "Precision": run_precision,
                "Recall": run_recall,
                "F1": run_f1,
                "ROC_AUC": run_roc_auc,
                "PR_AUC": run_pr_auc
            }
        )


        # ----------------------------------------------------
        # Save Run-level predictions
        # ----------------------------------------------------

        run_summary[
            "fold"
        ] = fold


        run_summary[
            "model"
        ] = model_name


        all_run_predictions.append(
            run_summary
        )


# ============================================================
# 15. Results DataFrames
# ============================================================

window_results_df = pd.DataFrame(
    window_results
)


run_results_df = pd.DataFrame(
    run_results
)


window_predictions_df = pd.concat(
    all_window_predictions,
    ignore_index=True
)


run_predictions_df = pd.concat(
    all_run_predictions,
    ignore_index=True
)


# ============================================================
# 16. Window-level summary
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


print("\n" + "=" * 110)
print("WINDOW-LEVEL FINAL SUMMARY")
print("=" * 110)

print(
    window_summary
)


# ============================================================
# 17. Run-level summary
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


print("\n" + "=" * 110)
print("RUN-LEVEL FINAL SUMMARY")
print("=" * 110)

print(
    run_summary
)


# ============================================================
# 18. Mean Run-level table
# ============================================================

mean_run_results = (
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
    .mean()
    .sort_values(
        by="F1",
        ascending=False
    )
)


print("\n" + "=" * 110)
print("RUN-LEVEL MODEL RANKING")
print("=" * 110)

print(
    mean_run_results
)


# ============================================================
# 19. Aggregated Window-level Confusion Matrices
# ============================================================

for model_name in [
    "Logistic Regression",
    "SVM-RBF",
    "Random Forest"
]:

    model_data = (
        window_predictions_df[
            window_predictions_df[
                "model"
            ]
            == model_name
        ]
    )


    cm = confusion_matrix(
        model_data[
            "true_label"
        ],
        model_data[
            "prediction"
        ]
    )


    print("\n" + "=" * 90)
    print(
        f"AGGREGATED WINDOW CONFUSION MATRIX - {model_name}"
    )
    print("=" * 90)

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
        f"{model_name} - Window Level"
    )


    fig.tight_layout()


# ============================================================
# 20. Aggregated Run-level Confusion Matrices
# ============================================================

for model_name in [
    "Logistic Regression",
    "SVM-RBF",
    "Random Forest"
]:

    model_data = (
        run_predictions_df[
            run_predictions_df[
                "model"
            ]
            == model_name
        ]
    )


    cm = confusion_matrix(
        model_data[
            "true_label"
        ],
        model_data[
            "prediction"
        ]
    )


    print("\n" + "=" * 90)
    print(
        f"AGGREGATED RUN CONFUSION MATRIX - {model_name}"
    )
    print("=" * 90)

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
        f"{model_name} - Run Level"
    )


    fig.tight_layout()


# ============================================================
# 21. Best final model
# ============================================================

best_model_name = (
    mean_run_results[
        "F1"
    ]
    .idxmax()
)


best_model_results = (
    mean_run_results
    .loc[
        best_model_name
    ]
)


print("\n" + "=" * 90)
print("FINAL CHAMPION MODEL")
print("=" * 90)


print(
    f"Model        : "
    f"{best_model_name}"
)


print(
    f"Mean Accuracy: "
    f"{best_model_results['Accuracy']:.6f}"
)


print(
    f"Mean Precision: "
    f"{best_model_results['Precision']:.6f}"
)


print(
    f"Mean Recall  : "
    f"{best_model_results['Recall']:.6f}"
)


print(
    f"Mean F1      : "
    f"{best_model_results['F1']:.6f}"
)


print(
    f"Mean ROC-AUC : "
    f"{best_model_results['ROC_AUC']:.6f}"
)


print(
    f"Mean PR-AUC  : "
    f"{best_model_results['PR_AUC']:.6f}"
)


# ============================================================
# 22. Save all result files
# ============================================================

window_results_df.to_csv(
    "data/final_window_metrics.csv",
    index=False
)


run_results_df.to_csv(
    "data/final_run_metrics.csv",
    index=False
)


window_predictions_df.to_csv(
    "data/final_window_predictions.csv",
    index=False
)


run_predictions_df.to_csv(
    "data/final_run_predictions.csv",
    index=False
)


mean_run_results.to_csv(
    "data/final_model_ranking.csv"
)


print("\n" + "=" * 90)
print("RESULT FILES SAVED")
print("=" * 90)


print(
    "data/final_window_metrics.csv"
)

print(
    "data/final_run_metrics.csv"
)

print(
    "data/final_window_predictions.csv"
)

print(
    "data/final_run_predictions.csv"
)

print(
    "data/final_model_ranking.csv"
)


# ============================================================
# 23. Show figures
# ============================================================

plt.show()