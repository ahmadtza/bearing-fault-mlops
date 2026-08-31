import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME",
    "bearing_condition_logistic",
)

CHAMPION_ALIAS = os.getenv(
    "CHAMPION_ALIAS",
    "champion",
)

EXPERIMENT_NAME = os.getenv(
    "BOOTSTRAP_EXPERIMENT_NAME",
    "Bearing Fault Detection - Bootstrap",
)

BASELINE_MODEL_DIR = Path(
    os.getenv(
        "BASELINE_MODEL_DIR",
        "models",
    )
)

MODEL_PATH = (
    BASELINE_MODEL_DIR
    / "bearing_logistic_pipeline.joblib"
)

FEATURES_PATH = (
    BASELINE_MODEL_DIR
    / "selected_features.json"
)

METADATA_PATH = (
    BASELINE_MODEL_DIR
    / "model_metadata.json"
)


# ============================================================
# 2. Helpers
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def validate_artifacts():
    required_paths = [
        MODEL_PATH,
        FEATURES_PATH,
        METADATA_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing baseline model artifacts: "
            + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)

    with FEATURES_PATH.open(
        encoding="utf-8"
    ) as file:
        selected_features = json.load(file)

    with METADATA_PATH.open(
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Baseline model has no predict() method."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Baseline model has no predict_proba() method."
        )

    if (
        metadata.get("selected_features")
        != selected_features
    ):
        raise ValueError(
            "Metadata selected_features does not match "
            "selected_features.json."
        )

    if (
        metadata.get("number_of_features")
        != len(selected_features)
    ):
        raise ValueError(
            "Metadata number_of_features is inconsistent."
        )

    if hasattr(model, "feature_names_in_"):
        if (
            list(model.feature_names_in_)
            != selected_features
        ):
            raise ValueError(
                "Model feature_names_in_ does not match "
                "the packaged feature contract."
            )

    return model, selected_features, metadata


def is_resource_missing(exc: MlflowException) -> bool:
    """
    Return True only when MLflow explicitly reports that the requested
    registry resource does not exist.

    Other MLflow errors (connection failures, server errors, permission
    problems, malformed requests, etc.) must propagate instead of being
    misinterpreted as an empty registry.
    """
    return (
        getattr(exc, "error_code", None)
        == "RESOURCE_DOES_NOT_EXIST"
    )


def get_existing_champion(client):
    try:
        return client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
        )

    except MlflowException as exc:
        if is_resource_missing(exc):
            return None
        raise


def registered_model_exists(client):
    try:
        client.get_registered_model(
            REGISTERED_MODEL_NAME
        )
        return True

    except MlflowException as exc:
        if is_resource_missing(exc):
            return False
        raise


# ============================================================
# 3. Main bootstrap
# ============================================================

def main():
    print_header(
        "BEARING MLFLOW MODEL BOOTSTRAP"
    )

    print(f"Tracking URI : {TRACKING_URI}")
    print(
        f"Model        : "
        f"{REGISTERED_MODEL_NAME}"
    )
    print(
        f"Alias        : "
        f"{CHAMPION_ALIAS}"
    )

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_registry_uri(
        TRACKING_URI
    )

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    # --------------------------------------------------------
    # Check existing champion first
    # --------------------------------------------------------

    champion = get_existing_champion(
        client
    )

    if champion is not None:
        print_header(
            "BOOTSTRAP NOT REQUIRED"
        )

        print(
            "Champion already exists."
        )
        print(
            f"Model   : {champion.name}"
        )
        print(
            f"Version : {champion.version}"
        )
        print(
            f"Alias   : {CHAMPION_ALIAS}"
        )

        print(
            "\nExisting registry state "
            "was left unchanged."
        )

        return

    # --------------------------------------------------------
    # Safety check
    #
    # If the registered model already exists but has no
    # champion, do NOT silently insert the packaged baseline.
    # This protects partially configured or production
    # registries from unexpected modification.
    # --------------------------------------------------------

    if registered_model_exists(
        client
    ):
        raise RuntimeError(
            f"Registered model "
            f"'{REGISTERED_MODEL_NAME}' already exists "
            f"but alias '@{CHAMPION_ALIAS}' is missing. "
            f"Bootstrap stopped to protect the existing "
            f"registry. Resolve the registry state manually."
        )

    # --------------------------------------------------------
    # Validate packaged model artifacts
    # --------------------------------------------------------

    print_header(
        "VALIDATING BASELINE ARTIFACTS"
    )

    (
        model,
        selected_features,
        metadata,
    ) = validate_artifacts()

    print(
        "[OK] Baseline model loaded"
    )

    print(
        f"[OK] Feature contract: "
        f"{len(selected_features)} features"
    )

    print(
        "[OK] Metadata validated"
    )

    # --------------------------------------------------------
    # Create / select bootstrap experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Log baseline model
    # --------------------------------------------------------

    print_header(
        "LOGGING BASELINE MODEL"
    )

    with mlflow.start_run(
        run_name="public_baseline_bootstrap"
    ) as run:

        mlflow.set_tags(
            {
                "project":
                    "bearing_fault_detection",

                "bootstrap":
                    "true",

                "model_type":
                    metadata.get(
                        "model_type",
                        "LogisticRegression",
                    ),

                "model_stage":
                    "public_baseline",

                "positive_class":
                    "Before_Anomalous",

                "negative_class":
                    "After_Normal",
            }
        )

        mlflow.log_params(
            {
                "sampling_frequency_hz":
                    metadata[
                        "sampling_frequency_hz"
                    ],

                "window_size_samples":
                    metadata[
                        "window_size_samples"
                    ],

                "step_size_samples":
                    metadata[
                        "step_size_samples"
                    ],

                "number_of_features":
                    metadata[
                        "number_of_features"
                    ],

                "run_level_threshold":
                    metadata[
                        "run_level_threshold"
                    ],
            }
        )

        mlflow.log_artifact(
            str(FEATURES_PATH),
            artifact_path="metadata",
        )

        mlflow.log_artifact(
            str(METADATA_PATH),
            artifact_path="metadata",
        )

        model_info = (
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
            )
        )

        print(
            f"Run ID    : "
            f"{run.info.run_id}"
        )

        print(
            f"Model URI : "
            f"{model_info.model_uri}"
        )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------

    print_header(
        "REGISTERING MODEL"
    )

    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
    )

    version = str(
        model_version.version
    )

    print(
        f"Registered model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version          : "
        f"{version}"
    )

    # --------------------------------------------------------
    # Add model description / tags
    # --------------------------------------------------------

    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            "Bearing condition classifier based on "
            "frequency-domain vibration features. "
            "0 = After maintenance / Normal, "
            "1 = Before maintenance / Anomalous."
        ),
    )

    version_tags = {
        "bootstrap": "true",
        "validation_status": "public_baseline",
        "sampling_frequency_hz": "12000",
        "window_size": "2048",
        "step_size": "1024",
        "number_of_features": str(
            len(selected_features)
        ),
    }

    for key, value in (
        version_tags.items()
    ):
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=version,
            key=key,
            value=value,
        )

    # --------------------------------------------------------
    # Assign champion alias
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    verified = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
        )
    )

    print_header(
        "BOOTSTRAP COMPLETED"
    )

    print(
        f"Champion model : "
        f"{verified.name}"
    )

    print(
        f"Champion version: "
        f"{verified.version}"
    )

    print(
        f"Champion URI   : "
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )

    print(
        "\nMLflow bootstrap completed "
        "successfully."
    )


if __name__ == "__main__":
    main()
