import h5py
import pandas as pd
import matplotlib.pyplot as plt


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
# 2. Load data
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    data = {}

    for feature_name, object_name in zip(
        feature_names,
        feature_objects
    ):
        data[feature_name] = refs[object_name][()].flatten()

    label_codes = refs["f"][()].flatten()


label_map = {
    1: "Before",
    2: "After"
}

labels = [
    label_map[int(code)]
    for code in label_codes
]


df = pd.DataFrame(data)
df.insert(0, "label", labels)


# ============================================================
# 3. Correlation matrix
# ============================================================

correlation = df[feature_names].corr()

print("\n" + "=" * 70)
print("HIGHLY CORRELATED FEATURE PAIRS")
print("=" * 70)

for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):

        value = correlation.iloc[i, j]

        if abs(value) > 0.90:

            print(
                f"{feature_names[i]:20s} <-> "
                f"{feature_names[j]:20s} : "
                f"{value:.4f}"
            )


# ============================================================
# 4. Correlation heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(11, 9))

image = ax.imshow(
    correlation,
    aspect="auto",
    vmin=-1,
    vmax=1
)

ax.set_xticks(range(len(feature_names)))
ax.set_yticks(range(len(feature_names)))

ax.set_xticklabels(
    feature_names,
    rotation=90
)

ax.set_yticklabels(feature_names)

ax.set_title("Feature Correlation Matrix")

fig.colorbar(
    image,
    ax=ax,
    label="Correlation"
)

fig.tight_layout()


# ============================================================
# 5. Boxplots for selected important features
# ============================================================

selected_features = [
    "ch1_RMS",
    "ch1_Kurtosis",
    "ch2_RMS",
    "ch3_SNR",
    "ch3_SINAD",
    "ch3_THD"
]


for feature in selected_features:

    fig, ax = plt.subplots(figsize=(7, 5))

    before = df.loc[
        df["label"] == "Before",
        feature
    ]

    after = df.loc[
        df["label"] == "After",
        feature
    ]

    ax.boxplot(
        [before, after],
        tick_labels=["Before", "After"],
        showfliers=False
    )

    ax.set_title(f"{feature}: Before vs After")
    ax.set_ylabel(feature)

    fig.tight_layout()


# ============================================================
# 6. Histograms
# ============================================================

for feature in [
    "ch1_RMS",
    "ch3_SNR"
]:

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df.loc[df["label"] == "Before", feature],
        bins=50,
        alpha=0.6,
        label="Before",
        density=True
    )

    ax.hist(
        df.loc[df["label"] == "After", feature],
        bins=50,
        alpha=0.6,
        label="After",
        density=True
    )

    ax.set_title(
        f"Distribution of {feature}"
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()


# ============================================================
# 7. Show all figures together
# ============================================================

plt.show()