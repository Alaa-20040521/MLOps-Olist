from pathlib import Path

import great_expectations as gx


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def validate_with_gx(
    file_name: str = "labeled_orders.parquet",
) -> dict:
    """
    Validate Olist data using Great Expectations.

    Returns:
        Dictionary containing validation success and results.
    """

    file_path = PROJECT_ROOT / "data" / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    context = gx.get_context()

    batch = context.data_sources.pandas_default.read_parquet(
        str(file_path)
    )

    expectations = [
        gx.expectations.ExpectColumnToExist(
            column="order_id"
        ),
        gx.expectations.ExpectColumnToExist(
            column="order_purchase_timestamp"
        ),
        gx.expectations.ExpectColumnToExist(
            column="item_count"
        ),
        gx.expectations.ExpectColumnToExist(
            column="total_item_price"
        ),
        gx.expectations.ExpectColumnToExist(
            column="customer_state"
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="item_count",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_item_price",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_freight_value",
            min_value=0,
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="late",
            value_set=[0, 1],
        ),
    ]

    results = []

    for expectation in expectations:
        result = batch.validate(expectation)
        results.append(result)

    success = all(
        result["success"]
        for result in results
    )

    if not success:
        failed = [
            result
            for result in results
            if not result["success"]
        ]

        raise ValueError(
            f"Great Expectations validation failed: "
            f"{len(failed)} expectation(s) failed."
        )

    return {
        "success": success,
        "expectations_checked": len(expectations),
        "file": file_name,
    }


if __name__ == "__main__":
    result = validate_with_gx()
    print("Great Expectations validation passed.")
    print(f"Expectations checked: {result['expectations_checked']}")
    print(f"File: {result['file']}")