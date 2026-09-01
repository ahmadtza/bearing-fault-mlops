import os
import sys

import h5py
import numpy as np
import requests


API_URL = os.getenv(
    "BEARING_API_URL",
    "https://127.0.0.1/predict",
)

API_USERNAME = os.getenv("BEARING_API_USERNAME")
API_PASSWORD = os.getenv("BEARING_API_PASSWORD")

DATA_FILE = os.getenv(
    "BEARING_DATA_FILE",
    "data/MachineData_export.mat",
)

REQUEST_TIMEOUT = 120

TEST_CASES = [
    {
        "run_id": 1,
        "expected_prediction": 1,
        "expected_label": "Before",
        "expected_condition": "Anomalous",
    },
    {
        "run_id": 25,
        "expected_prediction": 0,
        "expected_label": "After",
        "expected_condition": "Normal",
    },
]


def load_run(run_id: int):
    with h5py.File(DATA_FILE, "r") as file:
        ch1_all = np.asarray(file["ch1"], dtype=float)
        ch2_all = np.asarray(file["ch2"], dtype=float)
        ch3_all = np.asarray(file["ch3"], dtype=float)

        run_ids = (
            np.asarray(file["run_id"])
            .flatten()
            .astype(int)
        )

        labels = (
            np.asarray(file["label_code"])
            .flatten()
            .astype(int)
        )

        matching_indices = np.where(
            run_ids == run_id
        )[0]

        if len(matching_indices) == 0:
            raise ValueError(
                f"Run {run_id} was not found."
            )

        recording_index = int(
            matching_indices[0]
        )

        ch1 = ch1_all[recording_index]
        ch2 = ch2_all[recording_index]
        ch3 = ch3_all[recording_index]

        label_code = int(
            labels[recording_index]
        )

    return ch1, ch2, ch3, label_code


def validate_signal(
    name: str,
    signal: np.ndarray,
):
    if signal.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(signal) < 2048:
        raise ValueError(
            f"{name} contains fewer than "
            "2048 samples."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


def test_run(test_case):
    run_id = test_case["run_id"]

    expected_prediction = (
        test_case["expected_prediction"]
    )

    expected_label = (
        test_case["expected_label"]
    )

    expected_condition = (
        test_case["expected_condition"]
    )

    print()
    print("=" * 80)
    print(f"TESTING RUN {run_id}")
    print("=" * 80)

    ch1, ch2, ch3, label_code = (
        load_run(run_id)
    )

    validate_signal("ch1", ch1)
    validate_signal("ch2", ch2)
    validate_signal("ch3", ch3)

    if not (
        len(ch1)
        == len(ch2)
        == len(ch3)
    ):
        raise ValueError(
            "Channel lengths are not equal."
        )

    print(
        f"Samples/channel       : {len(ch1)}"
    )
    print(
        f"Dataset label_code    : {label_code}"
    )
    print(
        f"Expected prediction   : "
        f"{expected_prediction}"
    )
    print(
        f"Expected label        : "
        f"{expected_label}"
    )
    print(
        f"Expected condition    : "
        f"{expected_condition}"
    )

    if label_code != expected_prediction:
        raise RuntimeError(
            f"Ground-truth mismatch for Run "
            f"{run_id}: dataset={label_code}, "
            f"expected={expected_prediction}"
        )

    payload = {
        "ch1": ch1.tolist(),
        "ch2": ch2.tolist(),
        "ch3": ch3.tolist(),
    }

    response = requests.post(
        API_URL,
        json=payload,
        auth=(
            API_USERNAME,
            API_PASSWORD,
        ),
        timeout=REQUEST_TIMEOUT,
        verify=False,
    )

    print(
        f"HTTP status           : "
        f"{response.status_code}"
    )

    response.raise_for_status()

    result = response.json()

    prediction = int(
        result["prediction"]
    )

    label = result["label"]

    condition = result["condition"]

    anomaly_probability = float(
        result["mean_anomaly_probability"]
    )

    number_of_windows = int(
        result["number_of_windows"]
    )

    prediction_pass = (
        prediction
        == expected_prediction
    )

    label_pass = (
        label
        == expected_label
    )

    condition_pass = (
        condition
        == expected_condition
    )

    overall_pass = (
        prediction_pass
        and label_pass
        and condition_pass
    )

    print(
        f"Prediction            : "
        f"{prediction}"
    )
    print(
        f"Predicted label       : "
        f"{label}"
    )
    print(
        f"Predicted condition   : "
        f"{condition}"
    )
    print(
        f"Anomaly probability   : "
        f"{anomaly_probability:.6f}"
    )
    print(
        f"Number of windows     : "
        f"{number_of_windows}"
    )
    print(
        f"Result                : "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    return {
        "run_id": run_id,
        "expected_prediction": (
            expected_prediction
        ),
        "prediction": prediction,
        "label": label,
        "condition": condition,
        "anomaly_probability": (
            anomaly_probability
        ),
        "number_of_windows": (
            number_of_windows
        ),
        "passed": overall_pass,
    }


def main():
    print("=" * 80)
    print("BEARING FAULT API ACCEPTANCE TEST")
    print("=" * 80)

    if not API_USERNAME:
        print(
            "ERROR: BEARING_API_USERNAME "
            "is not set."
        )
        sys.exit(2)

    if not API_PASSWORD:
        print(
            "ERROR: BEARING_API_PASSWORD "
            "is not set."
        )
        sys.exit(2)

    print(f"API URL   : {API_URL}")
    print(f"Data file : {DATA_FILE}")

    results = []

    try:
        for test_case in TEST_CASES:
            result = test_run(
                test_case
            )
            results.append(result)

    except Exception as error:
        print()
        print("=" * 80)
        print("ACCEPTANCE TEST ERROR")
        print("=" * 80)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        sys.exit(1)

    print()
    print("=" * 80)
    print("ACCEPTANCE TEST SUMMARY")
    print("=" * 80)

    for result in results:
        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"Run {result['run_id']:2d} : "
            f"{status} | "
            f"prediction="
            f"{result['prediction']} | "
            f"condition="
            f"{result['condition']} | "
            f"p(anomaly)="
            f"{result['anomaly_probability']:.6f}"
        )

    all_passed = all(
        result["passed"]
        for result in results
    )

    print("-" * 80)

    if all_passed:
        print(
            "FINAL RESULT: PASS"
        )
        sys.exit(0)

    print(
        "FINAL RESULT: FAIL"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
