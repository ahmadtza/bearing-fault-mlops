import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

# Correlation threshold for redundancy removal
correlation_threshold = 0.95

# Candidate feature-set sizes
top_k_values = [
    5,
    10,
    15,
    20
]


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
# 3. Identify time-domain features
# ============================================================

time_suffixes = [
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
        column.endswith(suffix)
        for suffix in time_suffixes
    )
]


# ============================================================
# 4. Frequency-only features
# ============================================================

frequency_features = [
    column
    for column in all_feature_columns
    if column not in time_features
]


print("=" * 90)
print("FREQUENCY FEATURE SELECTION")
print("=" * 90)

print(
    f"\nTotal dataset shape: "
    f"{df.shape}"
)

print(
    f"All numerical features: "
    f"{len(all_feature_columns)}"
)

print(
    f"Time-domain features: "
    f"{len(time_features)}"
)

print(
    f"Frequency-domain features: "
    f"{len(frequency_features)}"
)


# ============================================================
# 5. X / y / groups
# ============================================================

X_frequency = df[
    frequency_features
].copy()

y = df["label"]

groups = df["run_id"]


# ============================================================
# 6. Correlation-based redundancy removal
#
# IMPORTANT:
# This is an unsupervised operation.
# No labels are used.
# ============================================================

correlation_matrix = (
    X_frequency
    .corr()
    .abs()
)


# Upper triangle only
upper_triangle = (
    correlation_matrix
    .where(
        np.triu(
            np.ones(
                correlation_matrix.shape
            ),
            k=1
        ).astype(bool)
    )
)


features_to_remove = [
    column
    for column in upper_triangle.columns
    if any(
        upper_triangle[column]
        > correlation_threshold
    )
]


reduced_frequency_features = [
    column
    for column in frequency_features
    if column not in features_to_remove
]


print("\n" + "=" * 90)
print("CORRELATION-BASED REDUNDANCY REMOVAL")
print("=" * 90)

print(
    f"\nCorrelation threshold: "
    f"{correlation_threshold}"
)

print(
    f"Original frequency features: "
    f"{len(frequency_features)}"
)

print(
    f"Removed correlated features: "
    f"{len(features_to_remove)}"
)

print(
    f"Remaining frequency features: "
    f"{len(reduced_frequency_features)}"
)


print("\nRemoved features:")

for feature in features_to_remove:
    print(" -", feature)


# ============================================================
# 7. Prepare Group CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)


# Generate splits only ONCE
# so every feature set sees identical train/test runs

splits = list(
    cv.split(
        X_frequency,
        y,
        groups
    )
)


# ============================================================
# 8. Rank features using Logistic coefficients
#
# IMPORTANT:
# We rank features separately inside each training fold.
# This avoids using test data during feature ranking.
# ============================================================

def rank_features_on_training_fold(
    X_train,
    y_train,
    candidate_features
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train[
            candidate_features
        ]
    )


    model = LogisticRegression(
        max_iter=3000,
        random_state=random_state
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    coefficients = pd.DataFrame(
        {
            "Feature":
                candidate_features,

            "Coefficient":
                model.coef_[0]
        }
    )


    coefficients[
        "Absolute_Coefficient"
    ] = (
        coefficients[
            "Coefficient"
        ].abs()
    )


    coefficients = (
        coefficients
        .sort_values(
            by="Absolute_Coefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )


    return coefficients


# ============================================================
# 9. Evaluation helper
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
    y_score
):

    return {
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
# 10. Containers
# ============================================================

window_results = []

run_results = []

feature_ranking_rows = []


# ============================================================
# 11. Cross-validation
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(
    splits,
    start=1
):

    print("\n" + "=" * 90)
    print(
        f"FOLD {fold}"
    )
    print("=" * 90)


    # --------------------------------------------------------
    # Fold data
    # --------------------------------------------------------

    train_df = df.iloc[
        train_idx
    ]

    test_df = df.iloc[
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


    train_runs = sorted(
        groups.iloc[
            train_idx
        ].unique()
    )

    test_runs = sorted(
        test_groups.unique()
    )


    print(
        "\nTrain runs:"
    )

    print(
        train_runs
    )


    print(
        "\nTest runs:"
    )

    print(
        test_runs
    )


    # --------------------------------------------------------
    # Rank features ONLY using training fold
    # --------------------------------------------------------

    ranking_df = rank_features_on_training_fold(
        train_df,
        y_train,
        reduced_frequency_features
    )


    ranking_df["Fold"] = fold


    feature_ranking_rows.append(
        ranking_df
    )


    print(
        "\nTop 10 features in this fold:"
    )

    print(
        ranking_df
        .head(10)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # 12. Candidate feature subsets
    # ========================================================

    ranked_features = (
        ranking_df["Feature"]
        .tolist()
    )


    candidate_sets = {}


    for k in top_k_values:

        actual_k = min(
            k,
            len(ranked_features)
        )

        candidate_sets[
            f"Top-{actual_k}"
        ] = (
            ranked_features[
                :actual_k
            ]
        )


    candidate_sets[
        "All Reduced"
    ] = (
        reduced_frequency_features
    )


    candidate_sets[
        "All Frequency"
    ] = (
        frequency_features
    )


    # ========================================================
    # 13. Evaluate each feature subset
    # ========================================================

    for feature_set_name, selected_features in candidate_sets.items():

        # ----------------------------------------------------
        # Select features
        # ----------------------------------------------------

        X_train = train_df[
            selected_features
        ]

        X_test = test_df[
            selected_features
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


        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        model = LogisticRegression(
            max_iter=3000,
            random_state=random_state
        )


        model.fit(
            X_train_scaled,
            y_train
        )


        y_pred = model.predict(
            X_test_scaled
        )


        y_score = (
            model.predict_proba(
                X_test_scaled
            )[:, 1]
        )


        # ====================================================
        # 14. Window-level metrics
        # ====================================================

        window_metrics = evaluate_predictions(
            y_test,
            y_pred,
            y_score
        )


        window_results.append(
            {
                "Fold":
                    fold,

                "Feature_Set":
                    feature_set_name,

                "Number_of_Features":
                    len(selected_features),

                **window_metrics
            }
        )


        # ====================================================
        # 15. Run-level aggregation
        # ====================================================

        run_df = pd.DataFrame(
            {
                "run_id":
                    test_groups.values,

                "true_label":
                    y_test.values,

                "score":
                    y_score
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
            >= 0.5
        ).astype(int)


        run_metrics = evaluate_predictions(
            run_summary[
                "true_label"
            ],
            run_summary[
                "prediction"
            ],
            run_summary[
                "mean_score"
            ]
        )


        run_results.append(
            {
                "Fold":
                    fold,

                "Feature_Set":
                    feature_set_name,

                "Number_of_Features":
                    len(selected_features),

                **run_metrics
            }
        )


# ============================================================
# 16. Combine feature rankings
# ============================================================

all_rankings = pd.concat(
    feature_ranking_rows,
    ignore_index=True
)


# Mean absolute coefficient across folds

mean_feature_ranking = (
    all_rankings
    .groupby(
        "Feature"
    )
    .agg(
        Mean_Absolute_Coefficient=(
            "Absolute_Coefficient",
            "mean"
        ),

        Std_Absolute_Coefficient=(
            "Absolute_Coefficient",
            "std"
        ),

        Mean_Coefficient=(
            "Coefficient",
            "mean"
        )
    )
    .sort_values(
        by="Mean_Absolute_Coefficient",
        ascending=False
    )
)


print("\n" + "=" * 100)
print("GLOBAL FREQUENCY FEATURE RANKING")
print("=" * 100)

print(
    mean_feature_ranking
    .head(25)
    .to_string()
)


# ============================================================
# 17. Convert metric results
# ============================================================

window_results_df = pd.DataFrame(
    window_results
)

run_results_df = pd.DataFrame(
    run_results
)


# ============================================================
# 18. Window-level summary
# ============================================================

window_summary = (
    window_results_df
    .groupby(
        [
            "Feature_Set",
            "Number_of_Features"
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
    .agg(
        ["mean", "std"]
    )
)


print("\n" + "=" * 110)
print("WINDOW-LEVEL FEATURE SELECTION SUMMARY")
print("=" * 110)

print(
    window_summary
)


# ============================================================
# 19. Run-level summary
# ============================================================

run_summary = (
    run_results_df
    .groupby(
        [
            "Feature_Set",
            "Number_of_Features"
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
    .agg(
        ["mean", "std"]
    )
)


print("\n" + "=" * 110)
print("RUN-LEVEL FEATURE SELECTION SUMMARY")
print("=" * 110)

print(
    run_summary
)


# ============================================================
# 20. Simple mean-only ranking table
# ============================================================

mean_run_results = (
    run_results_df
    .groupby(
        [
            "Feature_Set",
            "Number_of_Features"
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


mean_run_results = (
    mean_run_results
    .sort_values(
        by=[
            "F1",
            "Number_of_Features"
        ],
        ascending=[
            False,
            True
        ]
    )
)


print("\n" + "=" * 110)
print("RUN-LEVEL FEATURE SET RANKING")
print("=" * 110)

print(
    mean_run_results
    .to_string(
        index=False
    )
)


# ============================================================
# 21. Find smallest feature set with maximum mean F1
# ============================================================

best_f1 = (
    mean_run_results[
        "F1"
    ].max()
)


best_candidates = (
    mean_run_results[
        np.isclose(
            mean_run_results[
                "F1"
            ],
            best_f1
        )
    ]
    .sort_values(
        by="Number_of_Features"
    )
)


best_configuration = (
    best_candidates
    .iloc[0]
)


print("\n" + "=" * 90)
print("BEST COMPACT FREQUENCY FEATURE SET")
print("=" * 90)

print(
    f"Feature set       : "
    f"{best_configuration['Feature_Set']}"
)

print(
    f"Number of features: "
    f"{int(best_configuration['Number_of_Features'])}"
)

print(
    f"Mean Accuracy     : "
    f"{best_configuration['Accuracy']:.6f}"
)

print(
    f"Mean Precision    : "
    f"{best_configuration['Precision']:.6f}"
)

print(
    f"Mean Recall       : "
    f"{best_configuration['Recall']:.6f}"
)

print(
    f"Mean F1           : "
    f"{best_configuration['F1']:.6f}"
)

print(
    f"Mean ROC-AUC      : "
    f"{best_configuration['ROC_AUC']:.6f}"
)

print(
    f"Mean PR-AUC       : "
    f"{best_configuration['PR_AUC']:.6f}"
)


# ============================================================
# 22. Top global frequency features
# ============================================================

print("\n" + "=" * 90)
print("TOP 15 GLOBAL FREQUENCY FEATURES")
print("=" * 90)

top_15_global = (
    mean_feature_ranking
    .head(15)
)


print(
    top_15_global
    .to_string()
)