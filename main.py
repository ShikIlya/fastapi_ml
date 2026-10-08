from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from schemas import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn, TrainingConfigChurn, ErrorResponse
from dataset import read_churn_dataset, get_amount_rows, split_info_dataset, train_churn_model, split_churn_dataset
from model_storage import save_churn_model, load_churn_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError
from datetime import datetime, timezone
import pandas as pd
import logging
from training_history import save_training_record, load_training_history
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_data = load_churn_model()
    except FileNotFoundError:
        app.state.model_data = None

    yield

app = FastAPI(lifespan=lifespan)

PREDICT_ERROR_RESPONSES = {
    422: {
        "model": ErrorResponse,
        "description": "Ошибка входных данных или выполнения предсказания.",
        "content": {
            "application/json": {
                "examples": {
                    "missing_feature": {
                        "summary": "Отсутствует обязательный признак",
                        "value": {
                            "code": 422,
                            "message": "Некорректные данные запроса.",
                            "details": {
                                "errors": [
                                    {
                                        "location": ["body", 0, "monthly_fee"],
                                        "message": "Field required",
                                        "type": "missing",
                                    }
                                ]
                            },
                        },
                    },
                    "invalid_type": {
                        "summary": "Неверный тип значения",
                        "value": {
                            "code": 422,
                            "message": "Некорректные данные запроса.",
                            "details": {
                                "errors": [
                                    {
                                        "location": ["body", 0, "monthly_fee"],
                                        "message": (
                                            "Input should be a valid number, "
                                            "unable to parse string as a number"
                                        ),
                                        "type": "float_parsing",
                                    }
                                ]
                            },
                        },
                    },
                    "prediction_error": {
                        "summary": "Ошибка при вызове модели",
                        "value": {
                            "code": 422,
                            "message": (
                                "Не удалось выполнить предсказание. "
                                "Проверьте входные данные."
                            ),
                            "details": None,
                        },
                    },
                }
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "Модель отсутствует или не обучена.",
        "content": {
            "application/json": {
                "example": {
                    "code": 503,
                    "message": (
                        "Модель churn недоступна. "
                        "Сначала обучите её через POST /model/train."
                    ),
                    "details": None,
                }
            }
        },
    },
    500: {
        "model": ErrorResponse,
        "description": "Непредвиденная внутренняя ошибка.",
        "content": {
            "application/json": {
                "example": {
                    "code": 500,
                    "message": (
                        "Внутренняя ошибка сервиса. "
                        "Попробуйте повторить запрос позже."
                    ),
                    "details": None,
                }
            }
        },
    },
}

TRAIN_ERROR_RESPONSES = {
    422: {
        "model": ErrorResponse,
        "description": "Ошибка конфигурации обучения или данных.",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_model_type": {
                        "summary": "Неподдерживаемый тип модели",
                        "value": {
                            "code": 422,
                            "message": "Некорректные данные запроса.",
                            "details": {
                                "errors": [
                                    {
                                        "location": ["body", "model_type"],
                                        "message": (
                                            "Input should be 'logreg' "
                                            "or 'random_forest'"
                                        ),
                                        "type": "literal_error",
                                    }
                                ]
                            },
                        },
                    },
                    "training_error": {
                        "summary": "Ошибка гиперпараметров или обучения",
                        "value": {
                            "code": 422,
                            "message": (
                                "Не удалось обучить модель. "
                                "Проверьте гиперпараметры и соответствие "
                                "данных требованиям модели."
                            ),
                            "details": None,
                        },
                    },
                }
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "Датасет отсутствует или пуст.",
        "content": {
            "application/json": {
                "examples": {
                    "empty_dataset": {
                        "summary": "В CSV нет строк данных",
                        "value": {
                            "code": 503,
                            "message": "Датасет не содержит строк данных.",
                            "details": None,
                        },
                    },
                    "missing_dataset": {
                        "summary": "CSV-файл не найден",
                        "value": {
                            "code": 503,
                            "message": "Файл churn_dataset.csv не найден.",
                            "details": None,
                        },
                    },
                }
            }
        },
    },
    500: PREDICT_ERROR_RESPONSES[500],
}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    log_method = (
        logger.error
        if exc.status_code >= 500
        else logger.warning
    )

    log_method(
        "HTTP-ошибка: method=%s path=%s status=%s message=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    error = ErrorResponse(
        code=exc.status_code,
        message=str(exc.detail),
        details=None,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(mode="json"),
        headers=exc.headers,
    )

@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Необработанная ошибка: %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    error = ErrorResponse(
        code=500,
        message="Внутренняя ошибка сервиса. Попробуйте повторить запрос позже.",
        details=None,
    )

    return JSONResponse(
        status_code=500,
        content=error.model_dump(mode="json"),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = []

    for item in exc.errors():
        errors.append(
            {
                "location": list(item["loc"]),
                "message": item["msg"],
                "type": item["type"],
            }
        )

    logger.warning(
        "Ошибка валидации: method=%s path=%s errors=%s",
        request.method,
        request.url.path,
        errors,
    )

    error = ErrorResponse(
        code=422,
        message="Некорректные данные запроса.",
        details={"errors": errors},
    )

    return JSONResponse(
        status_code=422,
        content=error.model_dump(mode="json"),
    )

@app.post('/predict', responses=PREDICT_ERROR_RESPONSES)
def predict(payload: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    logger.info(
        "Получен запрос /predict: objects=%s",
        len(payload),
    )

    X = pd.DataFrame([client.model_dump() for client in payload])

    model_data = app.state.model_data

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

@app.get('/dataset/preview')
def preview(n: int = 5) -> list[DatasetRowChurn]:
    df = read_churn_dataset()
    df = get_amount_rows(df, n)

    return df.to_dict(orient='records')

@app.get('/dataset/info')
def info():
    df = read_churn_dataset()

    rows, columns = df.shape
    features = df.columns.tolist()
    distribution = df['churn'].value_counts().to_dict()

    return {
        'columns_count': columns,
        'rows_count': rows,
        'features': features,
        'distribution': distribution
    }

@app.get('/dataset/split-info')
def split_info():
    df = read_churn_dataset()

    return split_info_dataset(df)

@app.post('/model/train', responses=TRAIN_ERROR_RESPONSES)
def model_train(payload: TrainingConfigChurn):
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
    app.state.model_data = model_data

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

@app.get('/model/status')
def get_model_status():
    model_data = app.state.model_data

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

@app.get('/model/metrics')
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

@app.get('/model/schema')
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

@app.get('/health')
def get_health():
    model_available = app.state.model_data is not None

    try:
        read_churn_dataset()
        dataset_available = True
    except (OSError, ValueError, pd.errors.ParserError):
        dataset_available = False

    return {
        'model_available': model_available,
        'dataset_available': dataset_available
    }