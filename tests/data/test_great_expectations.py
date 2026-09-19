import pandas as pd
import great_expectations as gx

from src.validation.gx.great_expectations_validator import (
    validate_with_gx,
)


def test_great_expectations_validation_passes():
    result = validate_with_gx("labeled_orders.parquet")

    assert result["success"] is True
    assert result["expectations_checked"] == 9


def test_great_expectations_detects_invalid_late_value():
    df = pd.read_parquet("data/labeled_orders.parquet").copy()

    df.loc[0, "late"] = 2

    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_dataframe(df)

    result = batch.validate(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="late",
            value_set=[0, 1],
        )
    )

    assert result["success"] is False