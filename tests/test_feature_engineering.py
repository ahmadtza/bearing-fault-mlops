import numpy as np
import pytest

from feature_engineering import (
    SAMPLING_FREQUENCY,
    WINDOW_SIZE,
    STEP_SIZE,
    SELECTED_FEATURES,
    extract_frequency_features,
    extract_selected_features_from_window,
    extract_features_from_recording,
)


def create_sine_wave(
    frequency=1000.0,
    n_samples=WINDOW_SIZE,
    amplitude=1.0,
):
    """
    Create a deterministic sine wave for testing.
    """
    time = np.arange(n_samples) / SAMPLING_FREQUENCY

    return amplitude * np.sin(
        2.0 * np.pi * frequency * time
    )


def test_production_configuration():
    """
    Production signal-processing constants must not change accidentally.
    """
    assert SAMPLING_FREQUENCY == 12000.0
    assert WINDOW_SIZE == 2048
    assert STEP_SIZE == 1024
    assert len(SELECTED_FEATURES) == 10


def test_selected_feature_names():
    """
    Feature contract between feature engineering and ML model.
    """
    expected_features = [
        "ch3_SpectralCentroidHz",
        "ch3_BandEnergy_1000_2000",
        "ch3_RelEnergy_2000_4000",
        "ch3_SpectralFlatness",
        "ch3_BandEnergy_100_500",
        "ch3_RelEnergy_100_500",
        "ch3_BandEnergy_500_1000",
        "ch3_BandEnergy_2000_4000",
        "ch1_SpectralFlatness",
        "ch3_SpectralSpreadHz",
    ]

    assert SELECTED_FEATURES == expected_features


def test_dominant_frequency_for_sine_wave():
    """
    A 1000-Hz sine wave should have a dominant FFT frequency
    close to 1000 Hz.
    """
    signal = create_sine_wave(
        frequency=1000.0
    )

    features = extract_frequency_features(
        signal,
        "test",
    )

    dominant_frequency = features[
        "test_DominantFrequencyHz"
    ]

    frequency_resolution = (
        SAMPLING_FREQUENCY / WINDOW_SIZE
    )

    assert abs(
        dominant_frequency - 1000.0
    ) <= frequency_resolution


def test_selected_features_are_finite():
    """
    All production features must contain finite numerical values.
    """
    ch1 = create_sine_wave(500.0)
    ch2 = create_sine_wave(1000.0)
    ch3 = create_sine_wave(1500.0)

    features = (
        extract_selected_features_from_window(
            ch1,
            ch2,
            ch3,
        )
    )

    assert list(features.keys()) == SELECTED_FEATURES

    values = np.array(
        list(features.values()),
        dtype=float,
    )

    assert np.all(np.isfinite(values))


def test_recording_window_count():
    """
    70,000 samples must produce exactly 67 windows using
    WINDOW_SIZE=2048 and STEP_SIZE=1024.
    """
    n_samples = 70000

    ch1 = create_sine_wave(
        500.0,
        n_samples,
    )

    ch2 = create_sine_wave(
        1000.0,
        n_samples,
    )

    ch3 = create_sine_wave(
        1500.0,
        n_samples,
    )

    feature_df = extract_features_from_recording(
        ch1,
        ch2,
        ch3,
    )

    expected_windows = (
        (n_samples - WINDOW_SIZE)
        // STEP_SIZE
    ) + 1

    assert expected_windows == 67
    assert len(feature_df) == 67


def test_recording_output_columns():
    """
    Recording output must contain metadata plus the exact
    production feature contract.
    """
    n_samples = WINDOW_SIZE

    signal = create_sine_wave(
        1000.0,
        n_samples,
    )

    feature_df = extract_features_from_recording(
        signal,
        signal,
        signal,
    )

    expected_columns = [
        "window_id",
        "start_sample",
        "end_sample",
        *SELECTED_FEATURES,
    ]

    assert list(feature_df.columns) == expected_columns


def test_mismatched_channel_lengths_raise_error():
    """
    Three vibration channels must have identical lengths.
    """
    ch1 = np.zeros(3000)
    ch2 = np.zeros(3000)
    ch3 = np.zeros(2999)

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        extract_features_from_recording(
            ch1,
            ch2,
            ch3,
        )


def test_short_recording_raises_error():
    """
    Recording shorter than one analysis window must be rejected.
    """
    n_samples = WINDOW_SIZE - 1

    signal = np.zeros(n_samples)

    with pytest.raises(
        ValueError,
        match="At least 2048",
    ):
        extract_features_from_recording(
            signal,
            signal,
            signal,
        )


def test_feature_extraction_is_deterministic():
    """
    Identical input must generate identical production features.
    """
    signal = create_sine_wave(
        750.0
    )

    first = extract_selected_features_from_window(
        signal,
        signal,
        signal,
    )

    second = extract_selected_features_from_window(
        signal,
        signal,
        signal,
    )

    assert first.keys() == second.keys()

    for feature_name in SELECTED_FEATURES:
        assert np.isclose(
            first[feature_name],
            second[feature_name],
        )