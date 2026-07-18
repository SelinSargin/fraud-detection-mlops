from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "ok"
    assert response_data["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    response_data = response.json()

    assert "model_name" in response_data
    assert "threshold" in response_data
    assert "feature_columns" in response_data

    assert 0 <= response_data["threshold"] <= 1
    assert len(response_data["feature_columns"]) > 0



def test_predict_endpoint():
    model_info_response = client.get("/model-info")
    feature_columns = model_info_response.json()["feature_columns"]

    test_features = {
        feature_name: 0.0
        for feature_name in feature_columns
    }

    response = client.post(
        "/predict",
        json={
            "features": test_features
        }
    )

    assert response.status_code == 200

    response_data = response.json()

    assert "fraud_probability" in response_data
    assert "threshold" in response_data
    assert "prediction" in response_data
    assert "prediction_label" in response_data

    assert 0 <= response_data["fraud_probability"] <= 1
    assert response_data["prediction"] in [0, 1]
    assert response_data["prediction_label"] in ["normal", "fraud"]