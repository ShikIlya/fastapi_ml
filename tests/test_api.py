import pytest


def test_churn_pipeline(client, churn_df):
    # До обучения модели быть не должно.
    response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "trained": False,
        "trained_at": None,
        "metrics": None,
    }

    # Проверяем чтение тестового CSV через API.
    response = client.get("/dataset/info")

    assert response.status_code == 200
    assert response.json()["rows_count"] == len(churn_df)
    assert response.json()["columns_count"] == len(churn_df.columns)

    # Обучаем модель.
    config = {
        "model_type": "logreg",
        "hyperparameters": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        },
    }

    response = client.post("/model/train", json=config)

    assert response.status_code == 200, response.text

    metrics = response.json()

    assert set(metrics) == {"accuracy", "f1"}
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1

    # Проверяем состояние после обучения.
    response = client.get("/model/status")

    assert response.status_code == 200

    status = response.json()

    assert status["trained"] is True
    assert isinstance(status["trained_at"], int)
    assert status["metrics"] == metrics
    assert status["model_type"] == config["model_type"]
    assert status["hyperparameters"] == config["hyperparameters"]

    # Отправляем два объекта без целевой колонки churn.
    payload = (
        churn_df
        .drop(columns=["churn"])
        .head(2)
        .to_dict(orient="records")
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text

    predictions = response.json()

    assert len(predictions) == len(payload)

    for prediction in predictions:
        assert set(prediction) == {
            "churn",
            "probability_stay",
            "probability_churn",
        }

        assert prediction["churn"] in (0, 1)
        assert 0 <= prediction["probability_stay"] <= 1
        assert 0 <= prediction["probability_churn"] <= 1

        total_probability = (
            prediction["probability_stay"]
            + prediction["probability_churn"]
        )

        assert total_probability == pytest.approx(1.0)

    # Дополнительно проверяем сохранение истории.
    response = client.get("/model/metrics")

    assert response.status_code == 200

    history_response = response.json()

    assert history_response["latest_metrics"] == metrics
    assert len(history_response["history"]) == 1

    record = history_response["history"][0]

    assert record["model_type"] == config["model_type"]
    assert record["hyperparameters"] == config["hyperparameters"]
    assert record["metrics"] == metrics
    assert record["timestamp"] == status["trained_at"]

def test_predict_without_trained_model(client, churn_df):
    payload = (
        churn_df
        .drop(columns=["churn"])
        .head(1)
        .to_dict(orient="records")
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 503

    error = response.json()

    assert error == {
        "code": 503,
        "message": (
            "Модель churn недоступна. "
            "Сначала обучите её через POST /model/train."
        ),
        "details": None,
    }