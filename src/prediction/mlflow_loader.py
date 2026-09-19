from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "olist-late-delivery-model"
MODEL_ALIAS = "champion"

LOCAL_MODEL_ARTIFACT = (
    PROJECT_ROOT
    / "mlruns"
    / "1"
    / "models"
    / "m-5720cfe2ef43432da79cc26c23ca413a"
    / "artifacts"
)


def load_registered_model():
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    if not LOCAL_MODEL_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Registered model artifact not found: {LOCAL_MODEL_ARTIFACT}"
        )

    return mlflow.xgboost.load_model(
        str(LOCAL_MODEL_ARTIFACT)
    )