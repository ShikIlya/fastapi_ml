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