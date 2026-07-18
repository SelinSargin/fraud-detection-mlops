from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"

model = joblib.load(MODEL_PATH)  

with open(MODEL_INFO_PATH, "r") as f:
    model_info = json.load(f)

threshold = model_info["threshold"]
feature_columns = model_info["feature_columns"]


app = FastAPI(title="Fraud Detection API")


class TransactionRequest(BaseModel):
    features: dict[str, float]



@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True
    }



@app.post("/predict")
def predict(request: TransactionRequest):
    input_data = request.features

    row = pd.DataFrame([input_data])
    row = row[feature_columns]

    fraud_probability = model.predict_proba(row)[:, 1][0]
    prediction = int(fraud_probability >= threshold)

    return {
        "fraud_probability": float(fraud_probability),
        "threshold": threshold,
        "prediction": prediction,
        "prediction_label": "fraud" if prediction == 1 else "normal"
    }



@app.get("/model-info")
def get_model_info():
    return model_info