import pandas as pd


ID_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
]

LEAKAGE_COLUMNS = [
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
]

TARGET_COLUMN = "late"

RAW_DATETIME_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]

POST_PREDICTION_COLUMNS = [
    "review_count",
    "avg_review_score",
    "min_review_score",
    "max_review_score",
    "order_status",
]

NUMERIC_FEATURES = [
    "item_count",
    "total_item_price",
    "total_freight_value",
    "avg_item_price",
    "payment_count",
    "total_payment_value",
    "max_payment_installments",
    "unique_product_count",
    "unique_seller_count",
    "unique_category_count",
    "avg_latitude",
    "avg_longitude",
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "approval_delay_hours",
    "estimated_delivery_days_from_purchase",
]

CATEGORICAL_FEATURES = [
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
]

FINAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the six time-based features used during Task 2 training.
    """
    df = df.copy()

    purchase_time = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    approved_time = pd.to_datetime(
        df["order_approved_at"],
        errors="coerce",
    )

    estimated_delivery = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce",
    )

    df["purchase_year"] = purchase_time.dt.year
    df["purchase_month"] = purchase_time.dt.month
    df["purchase_dayofweek"] = purchase_time.dt.dayofweek
    df["purchase_hour"] = purchase_time.dt.hour

    df["approval_delay_hours"] = (
        (approved_time - purchase_time).dt.total_seconds() / 3600
    )

    df["estimated_delivery_days_from_purchase"] = (
        (estimated_delivery - purchase_time).dt.total_seconds() / 86400
    )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the exact feature preparation used in Task 2.

    This function:
    1. Creates time features.
    2. Removes IDs.
    3. Removes leakage columns.
    4. Removes target.
    5. Removes raw datetime columns.
    6. Removes post-prediction columns.
    7. Returns exactly the 21 features expected by the fitted preprocessor.
    """
    df = create_time_features(df)

    drop_columns = (
        ID_COLUMNS
        + LEAKAGE_COLUMNS
        + [TARGET_COLUMN]
        + RAW_DATETIME_COLUMNS
        + POST_PREDICTION_COLUMNS
    )

    existing_drop_columns = [
        column for column in drop_columns
        if column in df.columns
    ]

    X = df.drop(columns=existing_drop_columns)

    missing_features = [
        feature
        for feature in FINAL_FEATURES
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required engineered features: {missing_features}"
        )

    extra_features = [
        column
        for column in X.columns
        if column not in FINAL_FEATURES
    ]

    if extra_features:
        X = X.drop(columns=extra_features)

    return X[FINAL_FEATURES]