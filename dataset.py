import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from schemas import TrainingConfigChurn

NUMERIC_FEATURE_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]

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

def train_churn_model(X_train, y_train, config: TrainingConfigChurn):
    scaler = StandardScaler()
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', scaler, NUMERIC_FEATURE_COLUMNS),
            ('categorical', one_hot_encoder, CATEGORICAL_FEATURE_COLUMNS),
        ]
    )

    classifier = get_classifier(
        config.model_type,
        config.hyperparameters,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", column_transformer),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline

def get_classifier(model_type: str, hyperparameters: dict):
    if model_type == 'logreg':
        return LogisticRegression(**hyperparameters)

    if model_type == 'random_forest':
        return RandomForestClassifier(**hyperparameters)