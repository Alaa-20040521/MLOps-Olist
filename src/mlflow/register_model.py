from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_late_delivery_model.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_threshold.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "preprocessor.joblib"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    threshold_artifact = joblib.load(THRESHOLD_PATH)

    threshold = float(
        threshold_artifact["threshold"]
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "olist-late-delivery-prediction"
    )

    with mlflow.start_run(
        run_name="xgboost-task2-model"
    ) as run:

        mlflow.log_param(
            "model_type",
            "XGBClassifier",
        )

        mlflow.log_param(
            "model_version",
            "1.0.0",
        )

        mlflow.log_param(
            "prediction_threshold",
            threshold,
        )

        mlflow.log_param(
            "preprocessor",
            "ColumnTransformer",
        )

        mlflow.log_param(
            "transformed_features",
            17540,
        )

        # Metrics recorded from Task 2 validation results.
        mlflow.log_metric(
            "validation_accuracy",
            0.8997,
        )

        mlflow.log_metric(
            "validation_precision",
            0.2062,
        )

        mlflow.log_metric(
            "validation_recall",
            0.3079,
        )

        mlflow.log_metric(
            "validation_f1",
            0.2470,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            0.7558,
        )

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model_artifacts",
        )

        mlflow.log_artifact(
            str(THRESHOLD_PATH),
            artifact_path="model_artifacts",
        )

        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="model_artifacts",
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_late_delivery",
        )

        print("MLflow run completed.")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print("Experiment: olist-late-delivery-prediction")


if __name__ == "__main__":
    main()