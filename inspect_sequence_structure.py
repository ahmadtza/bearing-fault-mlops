import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/FeatureEntire.mat"


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
# 4. Decode labels
# ============================================================

label_map = {
    1: "Before",
    2: "After"
}


labels = np.array(
    [
        label_map[int(code)]
        for code in label_codes
    ]
)


# ============================================================
# 5. Create DataFrame
# ============================================================

df = pd.DataFrame(data)

df.insert(
    0,
    "label",
    labels
)


# Add original sequential row number
df.insert(
    0,
    "sample_index",
    np.arange(len(df))
)


# ============================================================
# 6. Basic information
# ============================================================

print("=" * 80)
print("SEQUENCE STRUCTURE ANALYSIS")
print("=" * 80)

print(f"\nTotal samples: {len(df)}")


print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
)


# ============================================================
# 7. First and last labels
# ============================================================

print("\n" + "=" * 80)
print("FIRST 20 LABELS")
print("=" * 80)

print(
    df[
        [
            "sample_index",
            "label"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("LAST 20 LABELS")
print("=" * 80)

print(
    df[
        [
            "sample_index",
            "label"
        ]
    ]
    .tail(20)
    .to_string(index=False)
)


# ============================================================
# 8. Find label transitions
# ============================================================

label_change = (
    df["label"]
    != df["label"].shift()
)


transition_indices = df.loc[
    label_change,
    "sample_index"
].to_numpy()


print("\n" + "=" * 80)
print("LABEL TRANSITIONS")
print("=" * 80)

print(
    f"Number of label blocks: "
    f"{len(transition_indices)}"
)

print(
    "\nStarting index of each block:"
)

print(
    transition_indices
)


# ============================================================
# 9. Construct continuous label blocks
# ============================================================

df["label_block"] = (
    label_change.cumsum()
)


block_summary = (
    df.groupby("label_block")
    .agg(
        Label=("label", "first"),
        Start_Index=("sample_index", "min"),
        End_Index=("sample_index", "max"),
        Number_of_Samples=("sample_index", "size")
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("CONTINUOUS LABEL BLOCKS")
print("=" * 80)

print(
    block_summary.to_string(
        index=False
    )
)


# ============================================================
# 10. Adjacent-sample absolute differences
# ============================================================

feature_df = df[
    all_features
]


absolute_difference = (
    feature_df
    .diff()
    .abs()
)


mean_adjacent_difference = (
    absolute_difference
    .mean()
)


feature_std = (
    feature_df
    .std()
)


relative_adjacent_difference = (
    mean_adjacent_difference
    / feature_std
)


adjacent_summary = pd.DataFrame(
    {
        "Mean_Adjacent_Absolute_Difference":
            mean_adjacent_difference,

        "Feature_Standard_Deviation":
            feature_std,

        "Relative_Adjacent_Difference":
            relative_adjacent_difference
    }
)


adjacent_summary = (
    adjacent_summary
    .sort_values(
        "Relative_Adjacent_Difference"
    )
)


print("\n" + "=" * 80)
print("ADJACENT SAMPLE SIMILARITY")
print("=" * 80)

print(
    adjacent_summary.to_string()
)


# ============================================================
# 11. Lag-1 autocorrelation
# ============================================================

autocorrelation_results = {}


for feature in all_features:

    autocorrelation_results[feature] = (
        df[feature]
        .autocorr(
            lag=1
        )
    )


autocorrelation_df = pd.DataFrame(
    {
        "Feature":
            list(
                autocorrelation_results.keys()
            ),

        "Lag1_Autocorrelation":
            list(
                autocorrelation_results.values()
            )
    }
)


autocorrelation_df = (
    autocorrelation_df
    .sort_values(
        "Lag1_Autocorrelation",
        ascending=False
    )
)


print("\n" + "=" * 80)
print("LAG-1 AUTOCORRELATION")
print("=" * 80)

print(
    autocorrelation_df
    .to_string(
        index=False
    )
)


# ============================================================
# 12. Selected features for sequential plots
# ============================================================

plot_features = [
    "ch1_RMS",
    "ch2_RMS",
    "ch3_SNR",
    "ch3_THD"
]


for feature in plot_features:

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["sample_index"],
        df[feature],
        linewidth=0.8
    )

    # Mark every label transition
    for transition in transition_indices[1:]:

        ax.axvline(
            x=transition,
            linestyle="--",
            linewidth=1.5
        )

    ax.set_xlabel(
        "Original Sample Index"
    )

    ax.set_ylabel(
        feature
    )

    ax.set_title(
        f"{feature} - Original Sequence"
    )

    fig.tight_layout()


# ============================================================
# 13. Rolling mean example
# ============================================================

rolling_window = 100


rolling_mean = (
    df["ch3_SNR"]
    .rolling(
        window=rolling_window
    )
    .mean()
)


fig, ax = plt.subplots(
    figsize=(12, 5)
)


ax.plot(
    df["sample_index"],
    df["ch3_SNR"],
    alpha=0.35,
    linewidth=0.7,
    label="Original"
)


ax.plot(
    df["sample_index"],
    rolling_mean,
    linewidth=2,
    label=f"Rolling Mean ({rolling_window})"
)


for transition in transition_indices[1:]:

    ax.axvline(
        x=transition,
        linestyle="--",
        linewidth=1.5
    )


ax.set_xlabel(
    "Original Sample Index"
)

ax.set_ylabel(
    "ch3_SNR"
)

ax.set_title(
    "ch3_SNR - Sequential Structure"
)

ax.legend()

fig.tight_layout()


# ============================================================
# 14. Show figures
# ============================================================

plt.show()