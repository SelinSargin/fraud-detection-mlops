from pathlib import Path
import json
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"


def test_model_can_be_loaded():
    assert MODEL_PATH.exists()

    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict_proba")


def test_model_info_is_valid():
    assert MODEL_INFO_PATH.exists()

    with open(MODEL_INFO_PATH, "r") as file:
        model_info = json.load(file)

    assert "model_name" in model_info
    assert "threshold" in model_info
    assert "feature_columns" in model_info

    assert 0 <= model_info["threshold"] <= 1
    assert len(model_info["feature_columns"]) > 0