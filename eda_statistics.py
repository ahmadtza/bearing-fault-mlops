import h5py
import pandas as pd


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
# 2. Load MATLAB data
# ============================================================

with h5py.File(file_path, "r") as file:

    refs = file["#refs#"]

    data = {}

    for feature_name, object_name in zip(
        feature_names,
        feature_objects
    ):
        data[feature_name] = (
            refs[object_name][()]
            .flatten()
        )

    label_codes = refs["f"][()].flatten()


# ============================================================
# 3. Convert labels
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
# 5. Mean value of each feature by class
# ============================================================

print("\n" + "=" * 80)
print("MEAN FEATURE VALUES BY CLASS")
print("=" * 80)

mean_by_class = (
    df.groupby("label")[feature_names]
    .mean()
    .T
)

print(mean_by_class)


# ============================================================
# 6. Standard deviation by class
# ============================================================

print("\n" + "=" * 80)
print("STANDARD DEVIATION BY CLASS")
print("=" * 80)

std_by_class = (
    df.groupby("label")[feature_names]
    .std()
    .T
)

print(std_by_class)


# ============================================================
# 7. Relative difference between class means
# ============================================================

comparison = mean_by_class.copy()

comparison["Absolute_Difference"] = (
    comparison["After"]
    - comparison["Before"]
).abs()

comparison["Relative_Difference_Percent"] = (
    comparison["Absolute_Difference"]
    /
    (
        (
            comparison["After"].abs()
            + comparison["Before"].abs()
        )
        / 2
    )
    * 100
)


comparison = comparison.sort_values(
    by="Relative_Difference_Percent",
    ascending=False
)


print("\n" + "=" * 80)
print("FEATURE DIFFERENCE: BEFORE vs AFTER")
print("=" * 80)

print(comparison)