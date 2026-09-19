from typing import Any

import pandas as pd

from src.features.feature_engineering import prepare_features
from src.prediction.artifact_loader import load_artifacts
from src.prediction.mlflow_loader import load_registered_model
from src.validation.input_validator import validate_input


class Predictor:
    """
    Inference pipeline for the Olist late-delivery model.

    The pipeline uses only fitted artifacts from Task 2.
    No training or fitting is performed.
    """

    def __init__(self) -> None:
        self.artifacts = load_artifacts()

        self.preprocessor = self.artifacts["preprocessor"]
        self.model = load_registered_model()
        self.threshold = self.artifacts["threshold"]
        self.feature_list = self.artifacts["feature_list"]

    def predict(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Generate predictions for one or more orders.

        Args:
            data: Raw input DataFrame.

        Returns:
            Dictionary containing predictions, probabilities,
            threshold, and model version.
        """

        if data.empty:
            raise ValueError("Input data cannot be empty.")

        # Step 1: Validate every input row.
        validated_rows = []

        for _, row in data.iterrows():
            validated_row = validate_input(row.to_dict())
            validated_rows.append(validated_row.iloc[0].to_dict())

        validated_data = pd.DataFrame(validated_rows)

        # Step 2: Reproduce Task 2 feature engineering.
        features = prepare_features(validated_data)

        # Step 3: Apply the fitted preprocessor.
        transformed_features = self.preprocessor.transform(features)

        # Step 4: Generate probability of late delivery.
        probabilities = self.model.predict_proba(
            transformed_features
        )[:, 1]

        # Step 5: Apply the saved Task 2 threshold.
        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "threshold": self.threshold,
            "model_version": "1.0.0",
        }