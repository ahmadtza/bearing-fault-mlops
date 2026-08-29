import h5py
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew


# ============================================================
# 1. Configuration
# ============================================================

file_path = "data/MachineData_export.mat"
output_file = "data/grouped_time_frequency_hz_features.csv"

sampling_frequency = 12000.0

window_size = 2048
step_size = 1024

epsilon = 1e-12


# ============================================================
# 2. Frequency information
# ============================================================

frequency_resolution = (
    sampling_frequency / window_size
)

nyquist_frequency = (
    sampling_frequency / 2
)


print("=" * 80)
print("SIGNAL / FFT CONFIGURATION")
print("=" * 80)

print(
    f"Sampling frequency : "
    f"{sampling_frequency:.2f} Hz"
)

print(
    f"Window size        : "
    f"{window_size} samples"
)

print(
    f"Frequency resolution: "
    f"{frequency_resolution:.6f} Hz"
)

print(
    f"Nyquist frequency  : "
    f"{nyquist_frequency:.2f} Hz"
)


# ============================================================
# 3. Load raw signals
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


print("\n" + "=" * 80)
print("RAW SIGNAL DATA")
print("=" * 80)

print("ch1:", ch1.shape)
print("ch2:", ch2.shape)
print("ch3:", ch3.shape)


# ============================================================
# 4. Time-domain functions
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(
            np.square(x)
        )
    )


def crest_factor(x):

    value = rms(x)

    if value < epsilon:
        return 0.0

    return (
        np.max(np.abs(x))
        / value
    )


def peak_to_peak(x):

    return (
        np.max(x)
        - np.min(x)
    )


# ============================================================
# 5. Time-domain features
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
# 6. Band-energy helper
# ============================================================

def calculate_band_energy(
    frequencies,
    power,
    low_frequency,
    high_frequency
):

    # Include the upper edge only for the final 6000-Hz band
    if high_frequency >= nyquist_frequency:

        mask = (
            (frequencies >= low_frequency)
            &
            (frequencies <= high_frequency)
        )

    else:

        mask = (
            (frequencies >= low_frequency)
            &
            (frequencies < high_frequency)
        )


    if not np.any(mask):
        return 0.0


    return np.sum(
        power[mask]
    )


# ============================================================
# 7. Frequency-domain features
# ============================================================

def extract_frequency_features(
    signal,
    prefix
):

    # --------------------------------------------------------
    # Remove mean / DC
    # --------------------------------------------------------

    centered_signal = (
        signal
        - np.mean(signal)
    )


    # --------------------------------------------------------
    # Hann window
    # --------------------------------------------------------

    hann_window = np.hanning(
        len(centered_signal)
    )

    windowed_signal = (
        centered_signal
        * hann_window
    )


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    spectrum = np.fft.rfft(
        windowed_signal
    )


    frequencies = np.fft.rfftfreq(
        len(windowed_signal),
        d=1.0 / sampling_frequency
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

    frequencies_no_dc = (
        frequencies[1:]
    )

    magnitude_no_dc = (
        magnitude[1:]
    )

    power_no_dc = (
        power[1:]
    )


    total_power = (
        np.sum(power_no_dc)
        + epsilon
    )


    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    dominant_index = np.argmax(
        magnitude_no_dc
    )


    dominant_frequency = (
        frequencies_no_dc[
            dominant_index
        ]
    )


    dominant_magnitude = (
        magnitude_no_dc[
            dominant_index
        ]
    )


    # --------------------------------------------------------
    # Spectral centroid in Hz
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(
            frequencies_no_dc
            * power_no_dc
        )
        / total_power
    )


    # --------------------------------------------------------
    # Spectral spread in Hz
    # --------------------------------------------------------

    spectral_spread = np.sqrt(
        np.sum(
            np.square(
                frequencies_no_dc
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

    spectral_energy = np.mean(
        power_no_dc
    )


    # --------------------------------------------------------
    # Spectral flatness
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
        np.mean(power_no_dc)
        + epsilon
    )


    spectral_flatness = (
        geometric_mean
        / arithmetic_mean
    )


    # --------------------------------------------------------
    # Frequency band energies
    # --------------------------------------------------------

    band_0_100 = calculate_band_energy(
        frequencies,
        power,
        0,
        100
    )

    band_100_500 = calculate_band_energy(
        frequencies,
        power,
        100,
        500
    )

    band_500_1000 = calculate_band_energy(
        frequencies,
        power,
        500,
        1000
    )

    band_1000_2000 = calculate_band_energy(
        frequencies,
        power,
        1000,
        2000
    )

    band_2000_4000 = calculate_band_energy(
        frequencies,
        power,
        2000,
        4000
    )

    band_4000_6000 = calculate_band_energy(
        frequencies,
        power,
        4000,
        6000
    )


    # --------------------------------------------------------
    # Relative band energies
    # --------------------------------------------------------

    band_total = (
        band_0_100
        + band_100_500
        + band_500_1000
        + band_1000_2000
        + band_2000_4000
        + band_4000_6000
        + epsilon
    )


    return {

        f"{prefix}_DominantFrequencyHz":
            dominant_frequency,

        f"{prefix}_DominantMagnitude":
            dominant_magnitude,

        f"{prefix}_SpectralCentroidHz":
            spectral_centroid,

        f"{prefix}_SpectralSpreadHz":
            spectral_spread,

        f"{prefix}_SpectralEntropy":
            spectral_entropy,

        f"{prefix}_SpectralEnergy":
            spectral_energy,

        f"{prefix}_SpectralFlatness":
            spectral_flatness,

        f"{prefix}_BandEnergy_0_100":
            band_0_100,

        f"{prefix}_BandEnergy_100_500":
            band_100_500,

        f"{prefix}_BandEnergy_500_1000":
            band_500_1000,

        f"{prefix}_BandEnergy_1000_2000":
            band_1000_2000,

        f"{prefix}_BandEnergy_2000_4000":
            band_2000_4000,

        f"{prefix}_BandEnergy_4000_6000":
            band_4000_6000,

        f"{prefix}_RelEnergy_0_100":
            band_0_100 / band_total,

        f"{prefix}_RelEnergy_100_500":
            band_100_500 / band_total,

        f"{prefix}_RelEnergy_500_1000":
            band_500_1000 / band_total,

        f"{prefix}_RelEnergy_1000_2000":
            band_1000_2000 / band_total,

        f"{prefix}_RelEnergy_2000_4000":
            band_2000_4000 / band_total,

        f"{prefix}_RelEnergy_4000_6000":
            band_4000_6000 / band_total
    }


# ============================================================
# 8. Complete channel feature extraction
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
# 9. Extract features from all recordings
# ============================================================

rows = []

number_of_recordings = ch1.shape[0]
number_of_samples = ch1.shape[1]


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
# 10. DataFrame
# ============================================================

df = pd.DataFrame(rows)


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
# 11. Feature columns
# ============================================================

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


# ============================================================
# 12. Summary
# ============================================================

print("\n" + "=" * 80)
print("TIME + FREQUENCY Hz FEATURE DATASET")
print("=" * 80)


print("\nShape:")
print(df.shape)


print("\nNumber of features:")
print(
    len(feature_columns)
)


print("\nClass distribution:")

print(
    df["condition"]
    .value_counts()
)


# ============================================================
# 13. Data-quality check
# ============================================================

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


print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)

print(
    f"NaN count: {nan_count}"
)

print(
    f"Inf count: {inf_count}"
)


# ============================================================
# 14. Example physical-frequency features
# ============================================================

example_columns = [

    "run_id",
    "condition",

    "ch1_DominantFrequencyHz",
    "ch1_SpectralCentroidHz",

    "ch2_DominantFrequencyHz",
    "ch2_SpectralCentroidHz",

    "ch3_DominantFrequencyHz",
    "ch3_SpectralCentroidHz"
]


print("\n" + "=" * 80)
print("EXAMPLE FREQUENCY FEATURES IN Hz")
print("=" * 80)

print(
    df[example_columns]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. Mean relative band energy by condition
# ============================================================

relative_energy_columns = [
    column
    for column in feature_columns
    if "RelEnergy" in column
]


print("\n" + "=" * 80)
print("MEAN RELATIVE BAND ENERGY BY CONDITION")
print("=" * 80)

print(
    df.groupby(
        "condition"
    )[relative_energy_columns]
    .mean()
    .T
)


# ============================================================
# 16. Save
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