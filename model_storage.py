import joblib
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "churn_model.joblib"
)

def save_churn_model(model_data):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_data, MODEL_PATH)

def load_churn_model():
    return joblib.load(MODEL_PATH)
