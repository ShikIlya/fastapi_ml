from fastapi import APIRouter, HTTPException, Query, Request
import logging
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime, timezone
from typing import Literal

from schemas import TrainingConfigChurn, ErrorResponse
from dataset import read_churn_dataset, split_churn_dataset
from ml.pipeline import train_churn_model
from model_storage import save_churn_model
from training_history import save_training_record, load_training_history
from api.responses import TRAIN_ERROR_RESPONSES

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post('/model/train', responses=TRAIN_ERROR_RESPONSES)
def model_train(payload: TrainingConfigChurn, request: Request):
    logger.info(
        "Обучение churn начато: model_type=%s hyperparameters=%s",
        payload.model_type,
        payload.hyperparameters,
    )

    try:
        df = read_churn_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    X_train, X_test, y_train, y_test = split_churn_dataset(df)

    try:
        pipeline = train_churn_model(X_train, y_train, payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Не удалось обучить модель. Проверьте гиперпараметры "
                "и соответствие данных требованиям модели."
            ),
        ) from exc

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    trained_at = int(datetime.now(timezone.utc).timestamp())

    model_data = {
        'pipeline': pipeline,
        'trained_at': trained_at,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
        },
        'model_type': payload.model_type,
        'hyperparameters': payload.hyperparameters
    }

    save_churn_model(model_data)
    request.app.state.model_data = model_data

    record = {
        'timestamp': trained_at,
        'model_type': payload.model_type,
        'hyperparameters': payload.hyperparameters,
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
        },
    }

    save_training_record(record)

    logger.info(
        "Обучение churn завершено: model_type=%s accuracy=%.4f f1=%.4f",
        payload.model_type,
        accuracy,
        f1,
    )

    return {
        'accuracy': accuracy,
        'f1': f1
    }

@router.get('/model/status')
def get_model_status(request: Request):
    model_data = request.app.state.model_data

    if model_data is None:
        return {
            'trained': False,
            'trained_at': None,
            'metrics': None,
        }

    return {
        'trained': True,
        'trained_at': model_data['trained_at'],
        'metrics': model_data['metrics'],
        'model_type': model_data['model_type'],
        'hyperparameters': model_data['hyperparameters']
    }

@router.get('/model/metrics')
def get_model_metrics(
        limit: int = Query(default=1, ge=1),
        model_type: Literal["logreg", "random_forest"] | None = None,
):
    try:
        history = load_training_history()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="История обучений отсутствует",
        ) from exc

    if not history:
        raise HTTPException(
            status_code=404,
            detail="История обучений отсутствует",
        )

    if model_type is not None:
        history = [
            record
            for record in history
            if record["model_type"] == model_type
        ]

    if not history:
        raise HTTPException(
            status_code=404,
            detail=(
                "История обучений для указанного типа модели отсутствует"
            ),
        )

    latest_records = history[-limit:][::-1]

    return {
        "latest_metrics": latest_records[0]["metrics"],
        "history": latest_records,
    }

@router.get('/model/schema')
def get_model_schema():
    return {
        "features": [
            {
                "name": "monthly_fee",
                "type": "float"
            },
            {
                "name": "usage_hours",
                "type": "float"
            },
            {
                "name": "support_requests",
                "type": "int"
            },
            {
                "name": "account_age_months",
                "type": "int"
            },
            {
                "name": "failed_payments",
                "type": "int"
            },
            {
                "name": "region",
                "type": "string"
            },
            {
                "name": "device_type",
                "type": "string"
            },
            {
                "name": "payment_method",
                "type": "string"
            },
            {
                "name": "autopay_enabled",
                "type": "int"
            }
        ]
    }