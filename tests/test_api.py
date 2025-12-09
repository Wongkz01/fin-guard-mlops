from fastapi.testclient import TestClient
from src.api.app import app

# Create a test client that mimics a real user
client = TestClient(app)

def test_health_check():
    """Test if the API is alive"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FinGuard AI Service is Running"}

def test_prediction_safe():
    """Test a known SAFE transaction"""
    # 29 zeros (likely safe)
    fake_features = [0.0] * 29
    
    response = client.post("/predict", json={"features": fake_features})
    
    # Check if the request succeeded
    assert response.status_code == 200
    
    # Check if the logic is sound
    json_data = response.json()
    assert "fraud_probability" in json_data
    assert "is_fraud" in json_data
    assert json_data["status"] == "SAFE"

def test_prediction_input_error():
    """Test what happens if we send WRONG data (only 1 feature instead of 29)"""
    response = client.post("/predict", json={"features": [1.0]})
    
    # Should fail with 400 Bad Request
    assert response.status_code == 400