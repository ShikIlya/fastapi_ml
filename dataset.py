import pandas as pd
from sklearn.model_selection import train_test_split

def read_churn_dataset():
    df = pd.read_csv('data/churn_dataset.csv')
    return df

def get_amount_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n)

def split_info_dataset():
    df = read_churn_dataset()

    X = df.drop(columns=['churn'])
    y = df['churn']

    numeric_features_columns = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
    ]
    categorical_features_columns = [
        "region",
        "device_type",
        "payment_method",
        "autopay_enabled"
    ]

    numeric_features = X[numeric_features_columns]
    categorical_features = X[categorical_features_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()

    train_count_rows = len(y_train)
    test_count_rows = len(y_test)

    return {
        'train': {
            'rows': train_count_rows,
            'churn_distribution': train_counts.to_dict(),
        },
        'test': {
            'rows': test_count_rows,
            'churn_distribution': test_counts.to_dict(),
        }
    }