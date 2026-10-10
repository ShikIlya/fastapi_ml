import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def read_churn_dataset():
    try:
        df = pd.read_csv('data/churn_dataset.csv')
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Файл churn_dataset.csv не найден."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "CSV-файл пуст: нет заголовков и данных."
        ) from exc

    if df.empty:
        raise ValueError("Датасет не содержит строк данных.")

    logger.info(
        "Датасет churn загружен: rows=%s columns=%s",
        len(df),
        len(df.columns),
    )

    return df

def get_amount_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n)

def split_churn_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test

def split_info_dataset(df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = split_churn_dataset(df)

    return {
        "train": {
            "rows": len(X_train),
            "churn_distribution": y_train.value_counts().to_dict(),
        },
        "test": {
            "rows": len(X_test),
            "churn_distribution": y_test.value_counts().to_dict(),
        },
    }