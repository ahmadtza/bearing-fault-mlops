import os
import sys
import h5py
import numpy as np
import requests


# ============================================================
# Configuration
# ============================================================

#API_URL = "http://127.0.0.1:8000/predict"

#API_URL = "http://127.0.0.1/predict"
API_URL = "https://127.0.0.1/predict"

API_USERNAME = os.getenv("BEARING_API_USERNAME")
API_PASSWORD = os.getenv("BEARING_API_PASSWORD")

DATA_FILE = "data/MachineData_export.mat"

REQUEST_TIMEOUT = 120


# ============================================================
# Load one run from HDF5 / MAT file
# ============================================================

def load_run(run_id):

    if run_id < 1 or run_id > 40:
        raise ValueError(
            "run_id must be between 1 and 40."
        )

    with h5py.File(DATA_FILE, "r") as file:

        # ----------------------------------------------------
        # Load channels
        # ----------------------------------------------------

        ch1_all = np.asarray(
            file["ch1"],
            dtype=float
        )

        ch2_all = np.asarray(
            file["ch2"],
            dtype=float
        )

        ch3_all = np.asarray(
            file["ch3"],
            dtype=float
        )


        # ----------------------------------------------------
        # Load run IDs and labels safely
        # ----------------------------------------------------

        run_ids = (
            np.asarray(
                file["run_id"]
            )
            .flatten()
            .astype(int)
        )

        labels = (
            np.asarray(
                file["label_code"]
            )
            .flatten()
            .astype(int)
        )


        # ----------------------------------------------------
        # Find requested Run
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Extract recording
        # ----------------------------------------------------

        ch1 = ch1_all[
            recording_index
        ]

        ch2 = ch2_all[
            recording_index
        ]

        ch3 = ch3_all[
            recording_index
        ]

        label_code = int(
            labels[
                recording_index
            ]
        )


    return (
        ch1,
        ch2,
        ch3,
        label_code
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Read run ID
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python test_api.py <run_id>"
        )

        print(
            "\nExample:"
        )

        print(
            "python test_api.py 1"
        )

        sys.exit(1)


    try:

        run_id = int(
            sys.argv[1]
        )

    except ValueError:

        print(
            "ERROR: run_id must be an integer."
        )

        sys.exit(1)


    print("=" * 80)
    print("BEARING FAULT API TEST")
    print("=" * 80)

    print(
        f"\nRun ID  : {run_id}"
    )

    print(
        f"API URL : {API_URL}"
    )


    # --------------------------------------------------------
    # Load raw recording
    # --------------------------------------------------------

    print(
        "\nLoading raw recording..."
    )

    try:

        ch1, ch2, ch3, label_code = (
            load_run(
                run_id
            )
        )

    except Exception as error:

        print(
            f"\nDATA LOADING ERROR: {error}"
        )

        sys.exit(1)


    print(
        "Raw recording loaded successfully."
    )


    print("\n" + "=" * 80)
    print("RAW SIGNAL INFORMATION")
    print("=" * 80)

    print(
        f"ch1 samples : {len(ch1)}"
    )

    print(
        f"ch2 samples : {len(ch2)}"
    )

    print(
        f"ch3 samples : {len(ch3)}"
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if label_code == 1:

        true_label = "Before"
        true_condition = "Anomalous"

    else:

        true_label = "After"
        true_condition = "Normal"


    print(
        f"True label  : {true_label}"
    )

    print(
        f"Condition   : {true_condition}"
    )


    # --------------------------------------------------------
    # Build JSON request
    # --------------------------------------------------------

    print(
        "\nPreparing JSON request..."
    )


    payload = {

        "ch1":
            ch1.tolist(),

        "ch2":
            ch2.tolist(),

        "ch3":
            ch3.tolist()
    }


    total_values = (
        len(ch1)
        + len(ch2)
        + len(ch3)
    )


    print(
        f"Total signal values: "
        f"{total_values:,}"
    )


    # --------------------------------------------------------
    # Send HTTP request
    # --------------------------------------------------------

    print(
        "\nSending request to FastAPI..."
    )


    try:

        response = requests.post(

            API_URL,
            json=payload,
            auth=(API_USERNAME, API_PASSWORD),
            timeout=120,
            verify="certs/localhost.crt"
        )

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Could not connect to FastAPI."
        )

        print(
            "Make sure Uvicorn is running:"
        )

        print(
            "uvicorn app:app --reload"
        )

        sys.exit(1)

    except requests.exceptions.Timeout:

        print(
            "\nERROR: API request timed out."
        )

        sys.exit(1)

    except requests.RequestException as error:

        print(
            f"\nHTTP ERROR: {error}"
        )

        sys.exit(1)


    # --------------------------------------------------------
    # HTTP information
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HTTP RESPONSE")
    print("=" * 80)


    print(
        f"Status code: "
        f"{response.status_code}"
    )


    if response.status_code != 200:

        print(
            "\nAPI returned an error:"
        )

        print(
            response.text
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    try:

        result = response.json()

    except ValueError:

        print(
            "\nERROR: API did not return valid JSON."
        )

        print(
            response.text
        )

        sys.exit(1)


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("API PREDICTION RESULT")
    print("=" * 80)


    print(
        f"Number of samples        : "
        f"{result['number_of_samples']}"
    )

    print(
        f"Number of windows        : "
        f"{result['number_of_windows']}"
    )

    print(
        f"Normal windows           : "
        f"{result['normal_windows']}"
    )

    print(
        f"Anomalous windows        : "
        f"{result['anomalous_windows']}"
    )

    print(
        f"Mean anomaly probability : "
        f"{result['mean_anomaly_probability']:.6f}"
    )

    print(
        f"Decision threshold       : "
        f"{result['threshold']:.6f}"
    )

    print(
        f"Prediction code          : "
        f"{result['prediction']}"
    )

    print(
        f"Predicted label          : "
        f"{result['label']}"
    )

    print(
        f"Predicted condition      : "
        f"{result['condition']}"
    )


    # --------------------------------------------------------
    # Compare with ground truth
    # --------------------------------------------------------

    correct = (
        result["prediction"]
        == label_code
    )


    print("\n" + "=" * 80)
    print("GROUND TRUTH VERIFICATION")
    print("=" * 80)


    print(
        f"True label          : "
        f"{true_label}"
    )

    print(
        f"True condition      : "
        f"{true_condition}"
    )

    print(
        f"Predicted label     : "
        f"{result['label']}"
    )

    print(
        f"Predicted condition : "
        f"{result['condition']}"
    )

    print(
        f"Correct             : "
        f"{correct}"
    )


    print("\n" + "=" * 80)

    if correct:

        print(
            "END-TO-END API TEST PASSED"
        )

    else:

        print(
            "END-TO-END API TEST FAILED"
        )

    print("=" * 80)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()