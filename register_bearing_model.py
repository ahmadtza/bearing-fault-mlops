import mlflow
from mlflow import MlflowClient


# ============================================================
# 1. Configuration
# ============================================================

TRACKING_URI = "http://127.0.0.1:5000"

EXPERIMENT_NAME = "Bearing Fault Detection"

REGISTERED_MODEL_NAME = "bearing_condition_logistic"


# ============================================================
# 2. Configure MLflow
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI,
    registry_uri=TRACKING_URI
)


print("=" * 80)
print("BEARING MODEL REGISTRATION")
print("=" * 80)

print(f"\nTracking URI : {mlflow.get_tracking_uri()}")
print(f"Registry URI : {mlflow.get_registry_uri()}")


# ============================================================
# 3. Find experiment
# ============================================================

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise RuntimeError(
        f"Experiment not found: {EXPERIMENT_NAME}"
    )


experiment_id = experiment.experiment_id

print(f"Experiment ID: {experiment_id}")


# ============================================================
# 4. Find logged models in this experiment
# ============================================================

logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment_id],
    filter_string="name = 'model'",
    order_by=[
        {
            "field_name": "creation_time",
            "ascending": False
        }
    ],
    output_format="list"
)


if not logged_models:
    raise RuntimeError(
        "No logged model named 'model' was found."
    )


# Newest logged model
latest_model = logged_models[0]

model_id = latest_model.model_id
source_run_id = latest_model.source_run_id

logged_model_uri = f"models:/{model_id}"


print("\n" + "=" * 80)
print("LATEST LOGGED MODEL")
print("=" * 80)

print(f"Model ID : {model_id}")
print(f"Run ID   : {source_run_id}")
print(f"URI      : {logged_model_uri}")


# ============================================================
# 5. Register model version
# ============================================================

model_version = mlflow.register_model(
    model_uri=logged_model_uri,
    name=REGISTERED_MODEL_NAME,
    tags={
        "project": "bearing_fault_detection",
        "model_type": "LogisticRegression",
        "feature_domain": "frequency",
        "decision_level": "run",
        "validation_status": "pending"
    }
)


version = str(model_version.version)


print("\n" + "=" * 80)
print("MODEL VERSION CREATED")
print("=" * 80)

print(f"Registered model : {REGISTERED_MODEL_NAME}")
print(f"Version          : {version}")
print(f"Status           : {model_version.status}")


# ============================================================
# 6. Registered-model description
# ============================================================

client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description=(
        "Bearing condition classifier based on selected "
        "frequency-domain vibration features. "
        "0 = After maintenance / Normal, "
        "1 = Before maintenance / Anomalous."
    )
)


# ============================================================
# 7. Model-version description
# ============================================================

client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=version,
    description=(
        "StandardScaler + LogisticRegression. "
        "Sampling frequency: 12000 Hz. "
        "Window size: 2048 samples. "
        "Step size: 1024 samples. "
        "10 selected frequency-domain features. "
        "Validated with 5-fold StratifiedGroupKFold at run level."
    )
)


# ============================================================
# 8. Additional version tags
# ============================================================

version_tags = {
    "validation_status": "pending",
    "sampling_frequency_hz": "12000",
    "window_size": "2048",
    "step_size": "1024",
    "number_of_features": "10",
    "cv_run_f1": "1.0"
}


for key, value in version_tags.items():

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key=key,
        value=value
    )


# ============================================================
# 9. Candidate alias
# ============================================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="candidate",
    version=version
)


# ============================================================
# 10. Verify
# ============================================================

registered_version = client.get_model_version(
    name=REGISTERED_MODEL_NAME,
    version=version
)


candidate_version = (
    client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias="candidate"
    )
)


print("\n" + "=" * 80)
print("REGISTRATION VERIFICATION")
print("=" * 80)

print(f"Model name      : {registered_version.name}")
print(f"Model version   : {registered_version.version}")
print(f"Status          : {registered_version.status}")
print(f"Source Run ID   : {registered_version.run_id}")
print(f"Source          : {registered_version.source}")
print(f"Candidate alias : Version {candidate_version.version}")


print("\n" + "=" * 80)
print("REGISTERED MODEL URIs")
print("=" * 80)

print(
    f"Version URI   : "
    f"models:/{REGISTERED_MODEL_NAME}/{version}"
)

print(
    f"Candidate URI : "
    f"models:/{REGISTERED_MODEL_NAME}@candidate"
)


print("\n" + "=" * 80)
print("MODEL REGISTRATION COMPLETED")
print("=" * 80)