from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_data():
    return {"message": "ml churn service is running"}