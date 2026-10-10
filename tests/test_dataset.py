import pandas as pd
import pytest
import numpy as np

from dataset import (
    read_churn_dataset,
    get_amount_rows,
    split_churn_dataset,
    split_info_dataset,
)
from ml.pipeline import train_churn_model
from schemas import TrainingConfigChurn


def test_get_amount_rows(churn_df):
    result = get_amount_rows(churn_df, 5)

    assert len(result) == 5
    pd.testing.assert_frame_equal(result, churn_df.head(5))


def test_split_churn_dataset(churn_df):
    X_train, X_test, y_train, y_test = split_churn_dataset(churn_df)

    assert len(X_train) == 32
    assert len(X_test) == 8
    assert len(y_train) == 32
    assert len(y_test) == 8

    assert "churn" not in X_train.columns
    assert "churn" not in X_test.columns

    assert y_train.name == "churn"
    assert y_test.name == "churn"

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)

    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(churn_df.index)

    assert y_train.value_counts().to_dict() == {0: 16, 1: 16}
    assert y_test.value_counts().to_dict() == {0: 4, 1: 4}


def test_split_is_reproducible(churn_df):
    first_split = split_churn_dataset(churn_df)
    second_split = split_churn_dataset(churn_df)

    for first, second in zip(first_split, second_split):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)


def test_split_info_dataset(churn_df):
    result = split_info_dataset(churn_df)

    assert result == {
        "train": {
            "rows": 32,
            "churn_distribution": {0: 16, 1: 16},
        },
        "test": {
            "rows": 8,
            "churn_distribution": {0: 4, 1: 4},
        },
    }

def test_read_churn_dataset(churn_df, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    csv_path = data_dir / "churn_dataset.csv"
    churn_df.to_csv(csv_path, index=False)

    monkeypatch.chdir(tmp_path)

    result = read_churn_dataset()

    pd.testing.assert_frame_equal(result, churn_df)


def test_read_churn_dataset_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="не найден"):
        read_churn_dataset()


def test_read_churn_dataset_empty_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    csv_path = data_dir / "churn_dataset.csv"
    csv_path.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="CSV-файл пуст"):
        read_churn_dataset()


def test_read_churn_dataset_without_rows(churn_df, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    csv_path = data_dir / "churn_dataset.csv"
    churn_df.head(0).to_csv(csv_path, index=False)

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="не содержит строк"):
        read_churn_dataset()

@pytest.mark.parametrize(
    "model_type, hyperparameters",
    [
        ("logreg", {"C": 1.0, "max_iter": 1000}),
        (
            "random_forest",
            {
                "n_estimators": 10,
                "max_depth": 3,
                "random_state": 42,
            },
        ),
    ],
)
def test_train_churn_model(churn_df, model_type, hyperparameters):
    X_train, X_test, y_train, _ = split_churn_dataset(churn_df)

    config = TrainingConfigChurn(
        model_type=model_type,
        hyperparameters=hyperparameters,
    )

    pipeline = train_churn_model(X_train, y_train, config)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    assert list(pipeline.named_steps) == [
        "preprocessing",
        "classifier",
    ]

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})

    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    for row in probabilities:
        assert row.sum() == pytest.approx(1.0)

@pytest.mark.parametrize(
    "model_type, hyperparameters",
    [
        (
            "logreg",
            {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
            },
        ),
        (
            "random_forest",
            {
                "n_estimators": 10,
                "max_depth": 3,
                "random_state": 42,
            },
        ),
    ],
)
def test_training_is_reproducible(
    churn_df,
    model_type,
    hyperparameters,
):
    X_train, X_test, y_train, _ = split_churn_dataset(churn_df)

    config = TrainingConfigChurn(
        model_type=model_type,
        hyperparameters=hyperparameters,
    )

    first_pipeline = train_churn_model(X_train, y_train, config)
    second_pipeline = train_churn_model(X_train, y_train, config)

    first_predictions = first_pipeline.predict(X_test)
    second_predictions = second_pipeline.predict(X_test)

    first_probabilities = first_pipeline.predict_proba(X_test)
    second_probabilities = second_pipeline.predict_proba(X_test)

    np.testing.assert_array_equal(
        first_predictions,
        second_predictions,
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
        rtol=1e-10,
        atol=1e-12,
    )