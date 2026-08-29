import h5py
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/MachineData_export.mat"

output_file = "data/grouped_time_frequency_features.csv"

window_size = 2048
step_size = 1024

epsilon = 1e-12


# ============================================================
# 2. Load raw signals
# ============================================================

with h5py.File(file_path, "r") as file:

    ch1 = np.array(file["ch1"])
    ch2 = np.array(file["ch2"])
    ch3 = np.array(file["ch3"])

    run_id = (
        np.array(file["run_id"])
        .flatten()
        .astype(int)
    )

    label_code = (
        np.array(file["label_code"])
        .flatten()
        .astype(int)
    )


print("=" * 80)
print("RAW SIGNAL DATA")
print("=" * 80)

print("ch1:", ch1.shape)
print("ch2:", ch2.shape)
print("ch3:", ch3.shape)


# ============================================================
# 3. Time-domain helper functions
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(
            np.square(x)
        )
    )


def crest_factor(x):

    rms_value = rms(x)

    if rms_value < epsilon:
        return 0.0

    return (
        np.max(np.abs(x))
        / rms_value
    )


def peak_to_peak(x):

    return (
        np.max(x)
        - np.min(x)
    )


# ============================================================
# 4. Time-domain feature extraction
# ============================================================

def extract_time_features(
    signal,
    prefix
):

    return {

        f"{prefix}_Mean":
            np.mean(signal),

        f"{prefix}_Std":
            np.std(signal),

        f"{prefix}_RMS":
            rms(signal),

        f"{prefix}_Kurtosis":
            kurtosis(
                signal,
                fisher=False,
                bias=False
            ),

        f"{prefix}_Skewness":
            skew(
                signal,
                bias=False
            ),

        f"{prefix}_CrestFactor":
            crest_factor(signal),

        f"{prefix}_PeakToPeak":
            peak_to_peak(signal)
    }


# ============================================================
# 5. Frequency-domain feature extraction
# ============================================================

def extract_frequency_features(
    signal,
    prefix
):

    # --------------------------------------------------------
    # Remove DC component
    # --------------------------------------------------------

    centered_signal = (
        signal
        - np.mean(signal)
    )


    # --------------------------------------------------------
    # Apply Hann window
    # --------------------------------------------------------

    hann_window = np.hanning(
        len(centered_signal)
    )

    windowed_signal = (
        centered_signal
        * hann_window
    )


    # --------------------------------------------------------
    # One-sided FFT
    # --------------------------------------------------------

    spectrum = np.fft.rfft(
        windowed_signal
    )


    magnitude = np.abs(
        spectrum
    )


    power = np.square(
        magnitude
    )


    # --------------------------------------------------------
    # Remove DC bin
    # --------------------------------------------------------

    if len(magnitude) > 1:

        magnitude_no_dc = (
            magnitude[1:]
        )

        power_no_dc = (
            power[1:]
        )

        bins = np.arange(
            1,
            len(magnitude)
        )

    else:

        magnitude_no_dc = magnitude

        power_no_dc = power

        bins = np.arange(
            len(magnitude)
        )


    total_power = (
        np.sum(power_no_dc)
        + epsilon
    )


    # --------------------------------------------------------
    # Dominant frequency bin
    # --------------------------------------------------------

    dominant_index = np.argmax(
        magnitude_no_dc
    )


    dominant_bin = (
        bins[dominant_index]
    )


    dominant_magnitude = (
        magnitude_no_dc[
            dominant_index
        ]
    )


    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(
            bins
            * power_no_dc
        )
        / total_power
    )


    # --------------------------------------------------------
    # Spectral spread
    # --------------------------------------------------------

    spectral_spread = np.sqrt(
        np.sum(
            np.square(
                bins
                - spectral_centroid
            )
            * power_no_dc
        )
        / total_power
    )


    # --------------------------------------------------------
    # Spectral entropy
    # --------------------------------------------------------

    normalized_power = (
        power_no_dc
        / total_power
    )


    spectral_entropy = (
        -np.sum(
            normalized_power
            * np.log2(
                normalized_power
                + epsilon
            )
        )
    )


    # Normalize entropy to approximately [0, 1]
    if len(normalized_power) > 1:

        spectral_entropy = (
            spectral_entropy
            / np.log2(
                len(normalized_power)
            )
        )


    # --------------------------------------------------------
    # Spectral energy
    # --------------------------------------------------------

    spectral_energy = (
        np.mean(
            power_no_dc
        )
    )


    # --------------------------------------------------------
    # Spectral flatness
    #
    # geometric mean / arithmetic mean
    # --------------------------------------------------------

    geometric_mean = np.exp(
        np.mean(
            np.log(
                power_no_dc
                + epsilon
            )
        )
    )


    arithmetic_mean = (
        np.mean(
            power_no_dc
        )
        + epsilon
    )


    spectral_flatness = (
        geometric_mean
        / arithmetic_mean
    )


    return {

        f"{prefix}_DominantBin":
            dominant_bin,

        f"{prefix}_DominantMagnitude":
            dominant_magnitude,

        f"{prefix}_SpectralCentroid":
            spectral_centroid,

        f"{prefix}_SpectralSpread":
            spectral_spread,

        f"{prefix}_SpectralEntropy":
            spectral_entropy,

        f"{prefix}_SpectralEnergy":
            spectral_energy,

        f"{prefix}_SpectralFlatness":
            spectral_flatness
    }


# ============================================================
# 6. Complete feature extraction for one channel
# ============================================================

def extract_channel_features(
    signal,
    prefix
):

    features = {}

    features.update(
        extract_time_features(
            signal,
            prefix
        )
    )

    features.update(
        extract_frequency_features(
            signal,
            prefix
        )
    )

    return features


# ============================================================
# 7. Window all recordings
# ============================================================

rows = []


number_of_recordings = (
    ch1.shape[0]
)

number_of_samples = (
    ch1.shape[1]
)


for recording_index in range(
    number_of_recordings
):

    current_run_id = (
        run_id[recording_index]
    )

    current_label = (
        label_code[recording_index]
    )

    window_id = 0


    for start_sample in range(
        0,
        number_of_samples
        - window_size
        + 1,
        step_size
    ):

        end_sample = (
            start_sample
            + window_size
        )

        window_id += 1


        window_ch1 = ch1[
            recording_index,
            start_sample:end_sample
        ]

        window_ch2 = ch2[
            recording_index,
            start_sample:end_sample
        ]

        window_ch3 = ch3[
            recording_index,
            start_sample:end_sample
        ]


        row = {

            "run_id":
                current_run_id,

            "window_id":
                window_id,

            "start_sample":
                start_sample,

            "end_sample":
                end_sample,

            "label":
                current_label
        }


        row.update(
            extract_channel_features(
                window_ch1,
                "ch1"
            )
        )

        row.update(
            extract_channel_features(
                window_ch2,
                "ch2"
            )
        )

        row.update(
            extract_channel_features(
                window_ch3,
                "ch3"
            )
        )


        rows.append(row)


# ============================================================
# 8. Create DataFrame
# ============================================================

df = pd.DataFrame(
    rows
)


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
# 9. Dataset summary
# ============================================================

print("\n" + "=" * 80)
print("TIME + FREQUENCY FEATURE DATASET")
print("=" * 80)

print("\nShape:")
print(df.shape)


print("\nNumber of feature columns:")

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


print(
    len(feature_columns)
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
# 10. Data quality
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)


nan_count = (
    df[feature_columns]
    .isnull()
    .sum()
    .sum()
)


inf_count = (
    np.isinf(
        df[feature_columns]
        .to_numpy()
    )
    .sum()
)


print(
    f"NaN count: {nan_count}"
)

print(
    f"Inf count: {inf_count}"
)


# ============================================================
# 11. Example frequency features
# ============================================================

frequency_feature_columns = [
    column
    for column in feature_columns
    if (
        "Dominant"
        in column
        or "Spectral"
        in column
    )
]


print("\n" + "=" * 80)
print("EXAMPLE FREQUENCY FEATURES")
print("=" * 80)

print(
    df[
        [
            "run_id",
            "window_id",
            "condition"
        ]
        + frequency_feature_columns[:10]
    ]
    .head()
    .to_string(
        index=False
    )
)


# ============================================================
# 12. Save result
# ============================================================

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