import h5py
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/MachineData_export.mat"

window_size = 2048
step_size = 1024


# ============================================================
# 2. Load raw signals
# ============================================================

with h5py.File(file_path, "r") as file:

    ch1 = np.array(file["ch1"])
    ch2 = np.array(file["ch2"])
    ch3 = np.array(file["ch3"])

    run_id = np.array(
        file["run_id"]
    ).flatten().astype(int)

    label_code = np.array(
        file["label_code"]
    ).flatten().astype(int)


# HDF5 already gives:
# (40, 70000)

print("=" * 80)
print("RAW DATA")
print("=" * 80)

print("ch1:", ch1.shape)
print("ch2:", ch2.shape)
print("ch3:", ch3.shape)


# ============================================================
# 3. Feature helper functions
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(
            np.square(x)
        )
    )


def crest_factor(x):

    denominator = rms(x)

    if denominator == 0:
        return 0.0

    return (
        np.max(np.abs(x))
        / denominator
    )


def peak_to_peak(x):

    return (
        np.max(x)
        - np.min(x)
    )


# ============================================================
# 4. Feature extraction for one channel
# ============================================================

def extract_channel_features(
    signal,
    prefix
):

    features = {}

    features[f"{prefix}_Mean"] = (
        np.mean(signal)
    )

    features[f"{prefix}_Std"] = (
        np.std(signal)
    )

    features[f"{prefix}_RMS"] = (
        rms(signal)
    )

    features[f"{prefix}_Kurtosis"] = (
        kurtosis(
            signal,
            fisher=False,
            bias=False
        )
    )

    features[f"{prefix}_Skewness"] = (
        skew(
            signal,
            bias=False
        )
    )

    features[f"{prefix}_CrestFactor"] = (
        crest_factor(signal)
    )

    features[f"{prefix}_PeakToPeak"] = (
        peak_to_peak(signal)
    )

    return features


# ============================================================
# 5. Window all recordings
# ============================================================

rows = []


for recording_index in range(
    ch1.shape[0]
):

    current_run_id = (
        run_id[recording_index]
    )

    current_label = (
        label_code[recording_index]
    )

    number_of_samples = (
        ch1.shape[1]
    )

    window_id = 0


    for start in range(
        0,
        number_of_samples - window_size + 1,
        step_size
    ):

        end = (
            start + window_size
        )

        window_id += 1


        # --------------------------------------------
        # Extract windows
        # --------------------------------------------

        window_ch1 = ch1[
            recording_index,
            start:end
        ]

        window_ch2 = ch2[
            recording_index,
            start:end
        ]

        window_ch3 = ch3[
            recording_index,
            start:end
        ]


        # --------------------------------------------
        # Metadata
        # --------------------------------------------

        row = {
            "run_id":
                current_run_id,

            "window_id":
                window_id,

            "start_sample":
                start,

            "end_sample":
                end,

            "label":
                current_label
        }


        # --------------------------------------------
        # Channel 1 features
        # --------------------------------------------

        row.update(
            extract_channel_features(
                window_ch1,
                "ch1"
            )
        )


        # --------------------------------------------
        # Channel 2 features
        # --------------------------------------------

        row.update(
            extract_channel_features(
                window_ch2,
                "ch2"
            )
        )


        # --------------------------------------------
        # Channel 3 features
        # --------------------------------------------

        row.update(
            extract_channel_features(
                window_ch3,
                "ch3"
            )
        )


        rows.append(row)


# ============================================================
# 6. Create DataFrame
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# 7. Label text
# ============================================================

df["condition"] = (
    df["label"]
    .map(
        {
            0: "After_Normal",
            1: "Before_Anomalous"
        }
    )
)


# ============================================================
# 8. Basic summary
# ============================================================

print("\n" + "=" * 80)
print("GROUPED FEATURE DATASET")
print("=" * 80)

print("\nShape:")
print(df.shape)


print("\nFirst rows:")
print(
    df.head()
)


print("\nWindows per run:")

print(
    df.groupby(
        "run_id"
    )
    .size()
)


print("\nClass distribution:")

print(
    df["condition"]
    .value_counts()
)


# ============================================================
# 9. Check missing values
# ============================================================

print("\nMissing values:")

print(
    df.isnull()
    .sum()
)


# ============================================================
# 10. Verify run-label consistency
# ============================================================

run_label_summary = (
    df.groupby(
        "run_id"
    )["condition"]
    .first()
)


print("\n" + "=" * 80)
print("RUN / LABEL SUMMARY")
print("=" * 80)

print(
    run_label_summary
)


# ============================================================
# 11. Save grouped feature dataset
# ============================================================

output_file = (
    "data/grouped_features.csv"
)


df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 80)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 80)

print(
    f"Saved to: {output_file}"
)