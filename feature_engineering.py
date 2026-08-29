import numpy as np
import pandas as pd


# ============================================================
# 1. Signal-processing configuration
# ============================================================

SAMPLING_FREQUENCY = 12000.0

WINDOW_SIZE = 2048

STEP_SIZE = 1024

EPSILON = 1e-12


# ============================================================
# 2. Final production feature set
# ============================================================

SELECTED_FEATURES = [
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
# 3. Band-energy helper
# ============================================================

def calculate_band_energy(
    frequencies,
    power,
    low_frequency,
    high_frequency
):

    nyquist_frequency = (
        SAMPLING_FREQUENCY / 2
    )


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


    return float(
        np.sum(
            power[mask]
        )
    )


# ============================================================
# 4. Frequency feature extraction for one channel
# ============================================================

def extract_frequency_features(
    signal,
    prefix
):

    signal = np.asarray(
        signal,
        dtype=float
    )


    # --------------------------------------------------------
    # Remove DC
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
        d=1.0 / SAMPLING_FREQUENCY
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
        + EPSILON
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
    # Spectral centroid
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(
            frequencies_no_dc
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
                + EPSILON
            )
        )
    )


    if len(
        normalized_power
    ) > 1:

        spectral_entropy = (
            spectral_entropy
            / np.log2(
                len(
                    normalized_power
                )
            )
        )


    # --------------------------------------------------------
    # Spectral energy
    # --------------------------------------------------------

    spectral_energy = float(
        np.mean(
            power_no_dc
        )
    )


    # --------------------------------------------------------
    # Spectral flatness
    # --------------------------------------------------------

    geometric_mean = np.exp(
        np.mean(
            np.log(
                power_no_dc
                + EPSILON
            )
        )
    )


    arithmetic_mean = (
        np.mean(
            power_no_dc
        )
        + EPSILON
    )


    spectral_flatness = (
        geometric_mean
        / arithmetic_mean
    )


    # --------------------------------------------------------
    # Band energies
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


    band_total = (
        band_0_100
        + band_100_500
        + band_500_1000
        + band_1000_2000
        + band_2000_4000
        + band_4000_6000
        + EPSILON
    )


    # --------------------------------------------------------
    # Return complete frequency feature dictionary
    # --------------------------------------------------------

    return {

        f"{prefix}_DominantFrequencyHz":
            float(
                dominant_frequency
            ),

        f"{prefix}_DominantMagnitude":
            float(
                dominant_magnitude
            ),

        f"{prefix}_SpectralCentroidHz":
            float(
                spectral_centroid
            ),

        f"{prefix}_SpectralSpreadHz":
            float(
                spectral_spread
            ),

        f"{prefix}_SpectralEntropy":
            float(
                spectral_entropy
            ),

        f"{prefix}_SpectralEnergy":
            float(
                spectral_energy
            ),

        f"{prefix}_SpectralFlatness":
            float(
                spectral_flatness
            ),

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
            band_0_100
            / band_total,

        f"{prefix}_RelEnergy_100_500":
            band_100_500
            / band_total,

        f"{prefix}_RelEnergy_500_1000":
            band_500_1000
            / band_total,

        f"{prefix}_RelEnergy_1000_2000":
            band_1000_2000
            / band_total,

        f"{prefix}_RelEnergy_2000_4000":
            band_2000_4000
            / band_total,

        f"{prefix}_RelEnergy_4000_6000":
            band_4000_6000
            / band_total
    }


# ============================================================
# 5. Extract production features from one window
# ============================================================

def extract_selected_features_from_window(
    ch1_window,
    ch2_window,
    ch3_window
):

    features = {}


    features.update(
        extract_frequency_features(
            ch1_window,
            "ch1"
        )
    )


    features.update(
        extract_frequency_features(
            ch2_window,
            "ch2"
        )
    )


    features.update(
        extract_frequency_features(
            ch3_window,
            "ch3"
        )
    )


    selected = {
        feature:
            features[feature]

        for feature
        in SELECTED_FEATURES
    }


    return selected


# ============================================================
# 6. Window complete recording
# ============================================================

def extract_features_from_recording(
    ch1,
    ch2,
    ch3
):

    ch1 = np.asarray(
        ch1,
        dtype=float
    ).flatten()

    ch2 = np.asarray(
        ch2,
        dtype=float
    ).flatten()

    ch3 = np.asarray(
        ch3,
        dtype=float
    ).flatten()


    if not (
        len(ch1)
        == len(ch2)
        == len(ch3)
    ):

        raise ValueError(
            "ch1, ch2 and ch3 must have "
            "the same number of samples."
        )


    if len(ch1) < WINDOW_SIZE:

        raise ValueError(
            f"Recording contains only "
            f"{len(ch1)} samples. "
            f"At least {WINDOW_SIZE} are required."
        )


    rows = []

    window_id = 0


    for start_sample in range(
        0,
        len(ch1)
        - WINDOW_SIZE
        + 1,
        STEP_SIZE
    ):

        end_sample = (
            start_sample
            + WINDOW_SIZE
        )

        window_id += 1


        ch1_window = ch1[
            start_sample:end_sample
        ]

        ch2_window = ch2[
            start_sample:end_sample
        ]

        ch3_window = ch3[
            start_sample:end_sample
        ]


        row = {

            "window_id":
                window_id,

            "start_sample":
                start_sample,

            "end_sample":
                end_sample
        }


        row.update(
            extract_selected_features_from_window(
                ch1_window,
                ch2_window,
                ch3_window
            )
        )


        rows.append(
            row
        )


    feature_df = pd.DataFrame(
        rows
    )


    return feature_df