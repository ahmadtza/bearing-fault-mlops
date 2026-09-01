import os
import sys

import numpy as np
import requests


API_URL = os.getenv(
    "BEARING_API_URL",
    "https://127.0.0.1/predict",
)

API_USERNAME = os.getenv(
    "BEARING_API_USERNAME"
)

API_PASSWORD = os.getenv(
    "BEARING_API_PASSWORD"
)

FIXTURE_FILE = os.getenv(
    "BEARING_FIXTURE_FILE",
    "tests/fixtures/bearing_acceptance_fixture.npz",
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


def load_fixture_run(
    fixture,
    run_id: int,
):
    prefix = f"run_{run_id}"

    ch1 = np.asarray(
        fixture[f"{prefix}_ch1"],
        dtype=float,
    )

    ch2 = np.asarray(
        fixture[f"{prefix}_ch2"],
        dtype=float,
    )

    ch3 = np.asarray(
        fixture[f"{prefix}_ch3"],
        dtype=float,
    )

    label = int(
        np.asarray(
            fixture[f"{prefix}_label"]
        ).flatten()[0]
    )

    return ch1, ch2, ch3, label


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


def run_case(
    fixture,
    test_case,
):
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
    print(f"TESTING FIXTURE RUN {run_id}")
    print("=" * 80)

    ch1, ch2, ch3, dataset_label = (
        load_fixture_run(
            fixture,
            run_id,
        )
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

    if dataset_label != expected_prediction:
        raise RuntimeError(
            f"Ground-truth mismatch for "
            f"Run {run_id}: "
            f"fixture={dataset_label}, "
            f"expected={expected_prediction}"
        )

    print(
        f"Samples/channel       : {len(ch1)}"
    )
    print(
        f"Fixture label         : {dataset_label}"
    )
    print(
        f"Expected prediction   : "
        f"{expected_prediction}"
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
        result[
            "mean_anomaly_probability"
        ]
    )

    number_of_windows = int(
        result["number_of_windows"]
    )

    passed = (
        prediction
        == expected_prediction
        and label
        == expected_label
        and condition
        == expected_condition
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
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "run_id": run_id,
        "prediction": prediction,
        "condition": condition,
        "anomaly_probability": (
            anomaly_probability
        ),
        "number_of_windows": (
            number_of_windows
        ),
        "passed": passed,
    }


def main():
    print("=" * 80)
    print(
        "BEARING API CI ACCEPTANCE TEST"
    )
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

    print(f"API URL      : {API_URL}")
    print(f"Fixture file : {FIXTURE_FILE}")

    results = []

    try:
        with np.load(
            FIXTURE_FILE,
            allow_pickle=False,
        ) as fixture:

            for test_case in TEST_CASES:
                results.append(
                    run_case(
                        fixture,
                        test_case,
                    )
                )

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
    print("CI ACCEPTANCE TEST SUMMARY")
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
        print("FINAL RESULT: PASS")
        sys.exit(0)

    print("FINAL RESULT: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
