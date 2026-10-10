from fastapi import APIRouter

from schemas import DatasetRowChurn
from dataset import read_churn_dataset, get_amount_rows, split_info_dataset

router = APIRouter()

@router.get('/dataset/preview')
def preview(n: int = 5) -> list[DatasetRowChurn]:
    df = read_churn_dataset()
    df = get_amount_rows(df, n)

    return df.to_dict(orient='records')

@router.get('/dataset/info')
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

@router.get('/dataset/split-info')
def split_info():
    df = read_churn_dataset()

    return split_info_dataset(df)