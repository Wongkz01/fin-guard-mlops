from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os

# --- 1. DEFINE MODEL ARCHITECTURE ---
# We must redefine the class so PyTorch knows the structure when loading
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetector, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

# --- 2. INITIALIZE API ---
app = FastAPI(title="FinGuard Fraud Detection API")

# --- 3. LOAD THE MODEL ---
MODEL_PATH = os.path.join("models", "fraud_model.pth")
input_dim = 29 # We know our data has 29 features (V1-V28 + Amount)

model = FraudDetector(input_dim)

try:
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode (turns off training specific layers)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you run Phase 2 successfully?")

# --- 4. DEFINE INPUT FORMAT (Pydantic) ---
# This ensures users send the correct data format
class Transaction(BaseModel):
    features: list[float] # Expecting a list of 29 numbers

# --- 5. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Check if input length is correct
    if len(transaction.features) != 29:
        raise HTTPException(status_code=400, detail=f"Expected 29 features, got {len(transaction.features)}")

    try:
        # Convert list to Tensor
        input_data = torch.tensor([transaction.features], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_data)
            fraud_probability = prediction.item()
            
        # Determine Label (Threshold 0.5)
        is_fraud = fraud_probability > 0.5
        
        return {
            "fraud_probability": round(fraud_probability, 4),
            "is_fraud": bool(is_fraud),
            "status": "ALERT" if is_fraud else "SAFE"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/")
def home():
    return {"message": "FinGuard AI Service is Running"}