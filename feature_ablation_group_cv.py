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


all_feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


# ============================================================
# 3. Define time-domain features
# ============================================================

time_keywords = [
    "_Mean",
    "_Std",
    "_RMS",
    "_Kurtosis",
    "_Skewness",
    "_CrestFactor",
    "_PeakToPeak"
]


time_features = [
    column
    for column in all_feature_columns
    if any(
        column.endswith(keyword)
        for keyword in time_keywords
    )
]


# ============================================================
# 4. Frequency-domain features
# ============================================================

frequency_features = [
    column
    for column in all_feature_columns
    if column not in time_features
]


# ============================================================
# 5. Feature sets
# ============================================================

feature_sets = {

    "Time Only":
        time_features,

    "Frequency Only":
        frequency_features,

    "Time + Frequency":
        all_feature_columns
}


print("=" * 80)
print("FEATURE SET INFORMATION")
print("=" * 80)


for name, features in feature_sets.items():

    print(
        f"{name:20s}: "
        f"{len(features)} features"
    )


# ============================================================
# 6. Target and groups
# ============================================================

y = df["label"]

groups = df["run_id"]


# ============================================================
# 7. Cross-validation
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)


# Generate splits ONCE
# so every feature set sees exactly the same runs.

dummy_X = df[
    all_feature_columns
]


splits = list(
    cv.split(
        dummy_X,
        y,
        groups
    )
)


# ============================================================
# 8. Results
# ============================================================

window_results = []

run_results = []


# ============================================================
# 9. Feature-set loop
# ============================================================

for feature_set_name, feature_columns in feature_sets.items():

    print("\n" + "=" * 80)
    print(
        f"FEATURE SET: {feature_set_name}"
    )
    print("=" * 80)

    X = df[
        feature_columns
    ]


    # ========================================================
    # Fold loop
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        splits,
        start=1
    ):

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

        test_groups = groups.iloc[
            test_idx
        ]


        # ----------------------------------------------------
        # Scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # ====================================================
        # Logistic Regression
        # ====================================================

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

        logistic_score = (
            logistic.predict_proba(
                X_test_scaled
            )[:, 1]
        )


        # ====================================================
        # Random Forest
        # ====================================================

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

        rf_score = (
            rf.predict_proba(
                X_test
            )[:, 1]
        )


        # ====================================================
        # SVM
        # ====================================================

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


        models = {

            "Logistic Regression": (
                logistic_pred,
                logistic_score,
                0.5
            ),

            "Random Forest": (
                rf_pred,
                rf_score,
                0.5
            ),

            "SVM-RBF": (
                svm_pred,
                svm_score,
                0.0
            )
        }


        # ====================================================
        # 10. Evaluate models
        # ====================================================

        for model_name, (
            prediction,
            score,
            run_threshold
        ) in models.items():

            # -----------------------------------------------
            # Window-level metrics
            # -----------------------------------------------

            window_results.append(
                {
                    "Feature_Set":
                        feature_set_name,

                    "Fold":
                        fold,

                    "Model":
                        model_name,

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


            # -----------------------------------------------
            # Run-level aggregation
            # -----------------------------------------------

            run_df = pd.DataFrame(
                {
                    "run_id":
                        test_groups.values,

                    "true_label":
                        y_test.values,

                    "score":
                        score
                }
            )


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


            run_summary[
                "prediction"
            ] = (
                run_summary[
                    "mean_score"
                ]
                >= run_threshold
            ).astype(int)


            run_results.append(
                {
                    "Feature_Set":
                        feature_set_name,

                    "Fold":
                        fold,

                    "Model":
                        model_name,

                    "Accuracy":
                        accuracy_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "prediction"
                            ]
                        ),

                    "Precision":
                        precision_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "prediction"
                            ],
                            zero_division=0
                        ),

                    "Recall":
                        recall_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "prediction"
                            ],
                            zero_division=0
                        ),

                    "F1":
                        f1_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "prediction"
                            ],
                            zero_division=0
                        ),

                    "ROC_AUC":
                        roc_auc_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "mean_score"
                            ]
                        ),

                    "PR_AUC":
                        average_precision_score(
                            run_summary[
                                "true_label"
                            ],
                            run_summary[
                                "mean_score"
                            ]
                        )
                }
            )


# ============================================================
# 11. DataFrames
# ============================================================

window_df = pd.DataFrame(
    window_results
)

run_df = pd.DataFrame(
    run_results
)


# ============================================================
# 12. Window-level summary
# ============================================================

window_summary = (
    window_df
    .groupby(
        [
            "Feature_Set",
            "Model"
        ]
    )[
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
    .reset_index()
)


print("\n" + "=" * 110)
print("WINDOW-LEVEL ABLATION SUMMARY")
print("=" * 110)

print(
    window_summary.to_string(
        index=False
    )
)


# ============================================================
# 13. Run-level summary
# ============================================================

run_summary = (
    run_df
    .groupby(
        [
            "Feature_Set",
            "Model"
        ]
    )[
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
    .reset_index()
)


print("\n" + "=" * 110)
print("RUN-LEVEL ABLATION SUMMARY")
print("=" * 110)

print(
    run_summary.to_string(
        index=False
    )
)


# ============================================================
# 14. Rank configurations by Run-level F1
# ============================================================

ranking = (
    run_summary
    .sort_values(
        by="F1",
        ascending=False
    )
)


print("\n" + "=" * 110)
print("RUN-LEVEL RANKING")
print("=" * 110)

print(
    ranking.to_string(
        index=False
    )
)


# ============================================================
# 15. Best configuration
# ============================================================

best = ranking.iloc[0]


print("\n" + "=" * 80)
print("BEST FEATURE / MODEL COMBINATION")
print("=" * 80)

print(
    f"Feature set : "
    f"{best['Feature_Set']}"
)

print(
    f"Model       : "
    f"{best['Model']}"
)

print(
    f"Accuracy    : "
    f"{best['Accuracy']:.6f}"
)

print(
    f"Recall      : "
    f"{best['Recall']:.6f}"
)

print(
    f"F1          : "
    f"{best['F1']:.6f}"
)

print(
    f"ROC-AUC     : "
    f"{best['ROC_AUC']:.6f}"
)

print(
    f"PR-AUC      : "
    f"{best['PR_AUC']:.6f}"
)