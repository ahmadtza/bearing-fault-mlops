import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"

random_state = 42

np.random.seed(random_state)
tf.random.set_seed(random_state)


# ============================================================
# 2. Feature definitions
# ============================================================

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


df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


# ============================================================
# 5. X and y
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


print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 6. Train / Validation / Test split
#
# 60% Train
# 20% Validation
# 20% Test
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
# 7. Keep ONLY Normal training samples
# ============================================================

X_train_normal = X_train.loc[
    y_train == 0
]


print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

print(
    f"Normal training samples: "
    f"{len(X_train_normal)}"
)

print(
    f"Validation samples      : "
    f"{len(X_val)}"
)

print(
    f"Final test samples      : "
    f"{len(X_test)}"
)


# ============================================================
# 8. Standardization
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
# 9. Build Autoencoder
# ============================================================

input_dim = X_train_normal_scaled.shape[1]


autoencoder = Sequential(
    [
        Input(shape=(input_dim,)),

        Dense(
            6,
            activation="relu"
        ),

        Dense(
            3,
            activation="relu",
            name="latent_space"
        ),

        Dense(
            6,
            activation="relu"
        ),

        Dense(
            input_dim,
            activation="linear"
        )
    ]
)


autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


print("\n" + "=" * 80)
print("AUTOENCODER ARCHITECTURE")
print("=" * 80)

autoencoder.summary()


# ============================================================
# 10. Early stopping
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 11. Train Autoencoder
#
# Only Normal samples are used as input AND target
# ============================================================

history = autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,

    epochs=150,

    batch_size=64,

    validation_split=0.20,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 12. Reconstruction-error function
# ============================================================

def reconstruction_error(
    model,
    X_data
):

    reconstructed = model.predict(
        X_data,
        verbose=0
    )

    errors = np.mean(
        np.square(
            X_data - reconstructed
        ),
        axis=1
    )

    return errors


# ============================================================
# 13. Validation anomaly scores
# ============================================================

validation_scores = reconstruction_error(
    autoencoder,
    X_val_scaled
)


# ============================================================
# 14. Threshold optimization function
# ============================================================

def optimize_threshold(
    y_true,
    scores,
    number_of_thresholds=500
):

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        number_of_thresholds
    )

    rows = []

    for threshold in thresholds:

        y_pred = (
            scores > threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
        )

    threshold_df = pd.DataFrame(
        rows
    )

    best_row = (
        threshold_df
        .sort_values(
            by="F1",
            ascending=False
        )
        .iloc[0]
    )

    return (
        best_row["Threshold"],
        best_row,
        threshold_df
    )


# ============================================================
# 15. Optimize threshold on Validation ONLY
# ============================================================

(
    best_threshold,
    best_validation,
    threshold_df
) = optimize_threshold(
    y_val,
    validation_scores
)


print("\n" + "=" * 80)
print("BEST VALIDATION THRESHOLD")
print("=" * 80)

print(
    f"Threshold : "
    f"{best_threshold:.6f}"
)

print(
    f"Precision : "
    f"{best_validation['Precision']:.6f}"
)

print(
    f"Recall    : "
    f"{best_validation['Recall']:.6f}"
)

print(
    f"F1        : "
    f"{best_validation['F1']:.6f}"
)


# ============================================================
# 16. Final TEST reconstruction scores
# ============================================================

test_scores = reconstruction_error(
    autoencoder,
    X_test_scaled
)


# ============================================================
# 17. Final Test predictions
# ============================================================

y_test_pred = (
    test_scores > best_threshold
).astype(int)


# ============================================================
# 18. Final Test metrics
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
# 19. Classification report
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
# 20. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)


print("\n" + "=" * 80)
print("FINAL CONFUSION MATRIX")
print("=" * 80)

print(cm)


# ============================================================
# 21. Training history
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    history.history["loss"],
    label="Training Loss"
)

ax.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "MSE Loss"
)

ax.set_title(
    "Autoencoder Training History"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 22. Confusion Matrix plot
# ============================================================

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
    "Autoencoder - Final Test"
)

fig.tight_layout()


# ============================================================
# 23. Validation threshold curves
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

ax.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1"
)


ax.axvline(
    x=best_threshold,
    linestyle="--",
    linewidth=2,
    label="Best Threshold"
)


ax.set_xlabel(
    "Reconstruction Error Threshold"
)

ax.set_ylabel(
    "Metric"
)

ax.set_title(
    "Autoencoder - Validation Threshold"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 24. Test reconstruction-error distribution
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    test_scores[
        y_test.values == 0
    ],
    bins=50,
    alpha=0.6,
    label="After (Normal)"
)


ax.hist(
    test_scores[
        y_test.values == 1
    ],
    bins=50,
    alpha=0.6,
    label="Before (Anomalous)"
)


ax.axvline(
    x=best_threshold,
    linestyle="--",
    linewidth=2,
    label="Optimized Threshold"
)


ax.set_xlabel(
    "Reconstruction Error"
)

ax.set_ylabel(
    "Frequency"
)

ax.set_title(
    "Autoencoder - Reconstruction Error Distribution"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 25. Show figures
# ============================================================

plt.show()