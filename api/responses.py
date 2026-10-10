from schemas import ErrorResponse

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