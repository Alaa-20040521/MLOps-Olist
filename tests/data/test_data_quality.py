import pandas as pd
import pytest

from src.validation.gx.data_quality import validate_data_quality


def test_labeled_data_quality_passes():
    df = pd.read_parquet("data/labeled_orders.parquet")

    validate_data_quality(df)


def test_invalid_late_value_fails():
    df = pd.read_parquet("data/labeled_orders.parquet").copy()

    df.loc[0, "late"] = 2

    with pytest.raises(
        ValueError,
        match="Column 'late' contains invalid values",
    ):
        validate_data_quality(df)


def test_negative_count_value_fails():
    df = pd.read_parquet("data/labeled_orders.parquet").copy()

    df.loc[0, "item_count"] = -1

    with pytest.raises(
        ValueError,
        match="Count columns contain negative values",
    ):
        validate_data_quality(df)