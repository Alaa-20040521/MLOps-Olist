from typing import Any

import pandas as pd


REQUIRED_INPUT_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "item_count",
    "total_item_price",
    "total_freight_value",
    "avg_item_price",
    "payment_count",
    "total_payment_value",
    "max_payment_installments",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "unique_product_count",
    "unique_seller_count",
    "unique_category_count",
    "avg_latitude",
    "avg_longitude",
]


NUMERIC_COLUMNS = [
    "item_count",
    "total_item_price",
    "total_freight_value",
    "avg_item_price",
    "payment_count",
    "total_payment_value",
    "max_payment_installments",
    "customer_zip_code_prefix",
    "unique_product_count",
    "unique_seller_count",
    "unique_category_count",
    "avg_latitude",
    "avg_longitude",
]


CATEGORICAL_COLUMNS = [
    "customer_city",
    "customer_state",
]


DATETIME_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]


def validate_input(data: dict[str, Any]) -> pd.DataFrame:
    """
    Validate raw input required for late-delivery prediction.

    The input contains only information available before prediction.
    """

    missing_columns = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if column not in data
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    unexpected_columns = [
        column
        for column in data
        if column not in REQUIRED_INPUT_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {unexpected_columns}"
        )

    df = pd.DataFrame([data])

    for column in NUMERIC_COLUMNS:
        value = df[column].iloc[0]

        if pd.isna(value):
            continue

        if not pd.api.types.is_number(value):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    for column in CATEGORICAL_COLUMNS:
        value = df[column].iloc[0]

        if pd.isna(value):
            continue

        if not isinstance(value, str):
            raise ValueError(
                f"Column '{column}' must be a string."
            )

    for column in DATETIME_COLUMNS:
        value = df[column].iloc[0]

        if pd.isna(value):
            continue

        parsed = pd.to_datetime(value, errors="coerce")

        if pd.isna(parsed):
            raise ValueError(
                f"Column '{column}' must contain a valid datetime."
            )

    return df[REQUIRED_INPUT_COLUMNS]