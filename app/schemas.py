from typing import List

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    order_purchase_timestamp: str
    order_approved_at: str
    order_estimated_delivery_date: str

    item_count: int = Field(ge=0)
    total_item_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)
    avg_item_price: float = Field(ge=0)

    payment_count: int = Field(ge=0)
    total_payment_value: float = Field(ge=0)
    max_payment_installments: int = Field(ge=0)

    customer_zip_code_prefix: int = Field(ge=0)

    customer_city: str
    customer_state: str

    unique_product_count: int = Field(ge=0)
    unique_seller_count: int = Field(ge=0)
    unique_category_count: int = Field(ge=0)

    avg_latitude: float
    avg_longitude: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str