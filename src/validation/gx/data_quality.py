import pandas as pd


REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
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


def validate_data_quality(df: pd.DataFrame) -> None:
    """
    Validate basic data-quality requirements for Olist data.

    The target column 'late' is optional because inference data
    does not contain the target.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    missing_rates = df[REQUIRED_COLUMNS].isna().mean()

    high_missing_columns = [
        column
        for column, rate in missing_rates.items()
        if rate > 0.50
    ]

    if high_missing_columns:
        raise ValueError(
            "Columns exceed 50% missing values: "
            f"{high_missing_columns}"
        )

    # Validate target only when it exists.
    if "late" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["late"]):
            raise ValueError(
                "Column 'late' must be numeric."
            )

        invalid_late_values = (
            set(df["late"].dropna().unique()) - {0, 1}
        )

        if invalid_late_values:
            raise ValueError(
                "Column 'late' contains invalid values: "
                f"{invalid_late_values}"
            )

    count_columns = [
        "item_count",
        "payment_count",
        "unique_product_count",
        "unique_seller_count",
        "unique_category_count",
    ]

    negative_count_columns = [
        column
        for column in count_columns
        if (df[column].dropna() < 0).any()
    ]

    if negative_count_columns:
        raise ValueError(
            "Count columns contain negative values: "
            f"{negative_count_columns}"
        )

    if (df["total_item_price"].dropna() < 0).any():
        raise ValueError(
            "Column 'total_item_price' contains negative values."
        )

    if (df["total_freight_value"].dropna() < 0).any():
        raise ValueError(
            "Column 'total_freight_value' contains negative values."
        )

    print("Data quality validation passed.")