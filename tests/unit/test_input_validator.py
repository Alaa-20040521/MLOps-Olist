import pytest

from src.validation.input_validator import validate_input


VALID_DATA = {
    "order_purchase_timestamp": "2018-01-01 10:00:00",
    "order_approved_at": "2018-01-01 10:15:00",
    "order_estimated_delivery_date": "2018-01-10 00:00:00",
    "item_count": 2,
    "total_item_price": 100.0,
    "total_freight_value": 20.0,
    "avg_item_price": 50.0,
    "payment_count": 1,
    "total_payment_value": 120.0,
    "max_payment_installments": 3,
    "customer_zip_code_prefix": 13000,
    "customer_city": "campinas",
    "customer_state": "SP",
    "unique_product_count": 2,
    "unique_seller_count": 1,
    "unique_category_count": 1,
    "avg_latitude": -22.9,
    "avg_longitude": -47.0,
}


def test_valid_input():
    result = validate_input(VALID_DATA)

    assert result.shape == (1, 18)
    assert list(result.columns) == list(VALID_DATA.keys())


def test_missing_required_column():
    data = VALID_DATA.copy()
    data.pop("item_count")

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_input(data)


def test_unexpected_column():
    data = VALID_DATA.copy()
    data["unexpected_column"] = 123

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_input(data)


def test_invalid_numeric_type():
    data = VALID_DATA.copy()
    data["item_count"] = "two"

    with pytest.raises(ValueError, match="must be numeric"):
        validate_input(data)


def test_invalid_datetime():
    data = VALID_DATA.copy()
    data["order_purchase_timestamp"] = "not-a-date"

    with pytest.raises(ValueError, match="valid datetime"):
        validate_input(data)