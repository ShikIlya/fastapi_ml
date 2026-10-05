from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import FeatureVectorChurn, DatasetRowChurn, PredictionResponseChurn
from dataset import read_churn_dataset, get_amount_rows, split_info_dataset, train_churn_model, split_churn_dataset
from model_storage import save_churn_model, load_churn_model
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import pandas as pd

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_data = load_churn_model()
    except FileNotFoundError:
        app.state.model_data = None

    yield

app = FastAPI(lifespan=lifespan)

@app.post('/predict')
def predict(payload: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    X = pd.DataFrame([client.model_dump() for client in payload])

    model_data = app.state.model_data

    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Модель churn недоступна. Сначала обучите её через POST /model/train.",
        )

    pipeline = model_data['pipeline']

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    results = []

    for prediction, probability in zip(predictions, probabilities):
        results.append(
            {
                'churn': prediction,
                'probability_stay': probability[0],
                'probability_churn': probability[1]
            }
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

    model_data = {
        'pipeline': pipeline,
        'trained_at': datetime.now(),
        'metrics': {
            'accuracy': accuracy,
            'f1': f1,
        }
    }

    save_churn_model(model_data)
    app.state.model_data = model_data

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
        'metrics': model_data['metrics']
    }