from pydantic import BaseModel, ConfigDict
from typing import Any, Literal

class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

class DatasetRowChurn(FeatureVectorChurn):
    churn: int

class PredictionResponseChurn(BaseModel):
    churn: int
    probability_stay: float
    probability_churn: float

class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"]
    hyperparameters: dict[str, Any]

class ErrorResponse(BaseModel):
    code: int
    message: str
    details: dict[str, Any] | None = None