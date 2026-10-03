from fastapi import FastAPI
from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import read_churn_dataset, get_amount_rows

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

