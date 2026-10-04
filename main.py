from fastapi import FastAPI
from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import read_churn_dataset, get_amount_rows, split_info_dataset, train_churn_model, split_churn_dataset
from sklearn.metrics import accuracy_score, f1_score

app = FastAPI()

@app.post('/predict')
def predict(payload: FeatureVectorChurn) -> FeatureVectorChurn:
    return payload

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

@app.post('/model/train')
def model_train():
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

    pipeline = train_churn_model(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'f1': f1
    }