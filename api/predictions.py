import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from sklearn.exceptions import NotFittedError

from schemas import (
    FeatureVectorChurn,
    PredictionResponseChurn,
    ErrorResponse,
)
from api.responses import PREDICT_ERROR_RESPONSES

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post('/predict', responses=PREDICT_ERROR_RESPONSES)
def predict(
    payload: list[FeatureVectorChurn],
    request: Request,
) -> list[PredictionResponseChurn]:
    logger.info(
        "Получен запрос /predict: objects=%s",
        len(payload),
    )

    X = pd.DataFrame([client.model_dump() for client in payload])

    model_data = request.app.state.model_data

    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Модель churn недоступна. Сначала обучите её через POST /model/train.",
        )

    pipeline = model_data['pipeline']

    try:
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
    except NotFittedError as exc:
        raise HTTPException(
            status_code=503,
            detail="Модель недоступна для предсказания. Необходимо обучить её.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Не удалось выполнить предсказание. Проверьте входные данные.",
        ) from exc

    results = []

    for prediction, probability in zip(predictions, probabilities):
        results.append(
            {
                'churn': prediction,
                'probability_stay': probability[0],
                'probability_churn': probability[1]
            }
        )

    logger.info(
        "Предсказание churn завершено: results=%s",
        len(results),
    )

    return results