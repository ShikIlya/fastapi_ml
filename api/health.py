from fastapi import APIRouter, Request
import pandas as pd

from dataset import read_churn_dataset

router = APIRouter()

@router.get('/health')
def get_health(request: Request):
    model_available = request.app.state.model_data is not None

    try:
        read_churn_dataset()
        dataset_available = True
    except (OSError, ValueError, pd.errors.ParserError):
        dataset_available = False

    return {
        'model_available': model_available,
        'dataset_available': dataset_available
    }