from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_INPUT = {
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"


def test_model_endpoint():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "olist-late-delivery-model"
    assert data["model_alias"] == "champion"
    assert data["model_type"] == "XGBClassifier"
    assert data["model_version"] == "1.0.0"
    assert data["threshold"] == 0.5500000000000002


def test_single_prediction():
    response = client.post(
        "/predict",
        json=VALID_INPUT,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["threshold"] == 0.5500000000000002
    assert data["model_version"] == "1.0.0"


def test_batch_prediction():
    batch = [
        VALID_INPUT,
        {
            **VALID_INPUT,
            "item_count": 1,
            "total_item_price": 50.0,
            "total_freight_value": 10.0,
            "avg_item_price": 50.0,
            "total_payment_value": 60.0,
            "max_payment_installments": 1,
            "customer_zip_code_prefix": 20000,
            "customer_city": "rio de janeiro",
            "customer_state": "RJ",
            "avg_latitude": -22.9,
            "avg_longitude": -43.2,
        },
    ]

    response = client.post(
        "/predict/batch",
        json=batch,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2
    assert all(
        prediction in [0, 1]
        for prediction in data["predictions"]
    )
    assert all(
        0 <= probability <= 1
        for probability in data["probabilities"]
    )
    assert data["threshold"] == 0.5500000000000002
    assert data["model_version"] == "1.0.0"


def test_empty_batch_is_rejected():
    response = client.post(
        "/predict/batch",
        json=[],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Batch input cannot be empty."


def test_invalid_input_is_rejected():
    invalid_input = {
        **VALID_INPUT,
        "item_count": -1,
    }

    response = client.post(
        "/predict",
        json=invalid_input,
    )

    assert response.status_code == 422