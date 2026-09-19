import pandas as pd

from src.prediction.predictor import Predictor
from src.validation.input_validator import REQUIRED_INPUT_COLUMNS


def test_single_prediction_pipeline():
    df = pd.read_parquet("data/test.parquet")

    input_df = df[REQUIRED_INPUT_COLUMNS].head(1)

    predictor = Predictor()

    result = predictor.predict(input_df)

    assert "predictions" in result
    assert "probabilities" in result
    assert "threshold" in result
    assert "model_version" in result

    assert len(result["predictions"]) == 1
    assert len(result["probabilities"]) == 1

    assert result["predictions"][0] in [0, 1]
    assert 0.0 <= result["probabilities"][0] <= 1.0

    assert result["threshold"] == 0.5500000000000002
    assert result["model_version"] == "1.0.0"


def test_batch_prediction_pipeline():
    df = pd.read_parquet("data/test.parquet")

    input_df = df[REQUIRED_INPUT_COLUMNS].head(5)

    predictor = Predictor()

    result = predictor.predict(input_df)

    assert len(result["predictions"]) == 5
    assert len(result["probabilities"]) == 5

    assert all(
        prediction in [0, 1]
        for prediction in result["predictions"]
    )

    assert all(
        0.0 <= probability <= 1.0
        for probability in result["probabilities"]
    )

    assert result["threshold"] == 0.5500000000000002
    assert result["model_version"] == "1.0.0"