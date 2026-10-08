import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app
import model_storage
import training_history

@pytest.fixture
def churn_df():
    rows = []

    for i in range(40):
        churn = i % 2

        rows.append({
            "monthly_fee": 30.0 + i,
            "usage_hours": 5.0 + i % 10,
            "support_requests": churn + i % 3,
            "account_age_months": 1 + i,
            "failed_payments": churn,
            "region": "north" if i % 3 == 0 else "south",
            "device_type": "mobile" if i % 3 == 0 else "desktop",
            "payment_method": "card" if i % 3 == 0 else "bank",
            "autopay_enabled": 1 - churn,
            "churn": churn,
        })

    return pd.DataFrame(rows)

@pytest.fixture
def client(churn_df, tmp_path, monkeypatch):
    # Создаём отдельный CSV для каждого теста.
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    churn_df.to_csv(
        data_dir / "churn_dataset.csv",
        index=False,
    )

    # read_churn_dataset() использует относительный путь.
    monkeypatch.chdir(tmp_path)

    # Перенаправляем сохранение модели и истории.
    monkeypatch.setattr(
        model_storage,
        "MODEL_PATH",
        tmp_path / "models" / "churn_model.joblib",
    )
    monkeypatch.setattr(
        training_history,
        "HISTORY_PATH",
        data_dir / "training_history.json",
    )

    with TestClient(app) as test_client:
        yield test_client