import pandas as pd

def read_churn_dataset():
    df = pd.read_csv('data/churn_dataset.csv')
    return df

def get_amount_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n)
