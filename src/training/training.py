import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sqlite3
import mlflow
import mlflow.pytorch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION (The "Control Panel") ---
# We use relative paths so this works on any machine
DB_PATH = os.path.join("data", "fraud_data.db")
MODEL_SAVE_PATH = os.path.join("models", "fraud_model.pth")

# Hyperparameters (The "Knobs" we turn to improve the model)
EPOCHS = 5              # How many times the model sees the full dataset
LEARNING_RATE = 0.001   # How fast the model learns
BATCH_SIZE = 64         # How many rows it processes at once
TEST_SIZE = 0.2         # 20% of data is kept secret for testing

# --- 2. DEFINE THE PYTORCH MODEL (The "Brain") ---
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetector, self).__init__()
        
        # Layer 1: Takes input (29 features) -> Compresses to 16 neurons
        self.layer_1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU() # Activation function (adds non-linearity)
        
        # Layer 2: Takes 16 neurons -> Outputs 1 score (0 to 1)
        self.layer_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid() # Squashes output between 0 (Legit) and 1 (Fraud)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.sigmoid(x)
        return x

# --- 3. DATA LOADING (The "Feeder") ---
def load_data():
    """Reads from SQL, separates Features (X) from Target (y)"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Did you run Phase 1?")
        
    print(f"Loading data from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()

    # Preprocessing
    # We drop 'Class' because it's the answer key.
    # We drop 'Time' because in this synthetic set it's just a counter, not useful.
    X = df.drop(['Class', 'Time'], axis=1).values
    y = df['Class'].values

    return X, y

# --- 4. THE TRAINING ENGINE ---
def train_pipeline():
    # A. Set the Experiment Name in MLflow
    # This creates a folder in 'mlruns/' to store our results
    mlflow.set_experiment("FinGuard_Fraud_Detection")
    
    # B. Start a Tracking Run
    with mlflow.start_run():
        print("--- Starting Training Pipeline ---")
        
        # 1. Prepare Data
        X, y = load_data()
        
        # Standardize Data (Important for Neural Networks!)
        # This makes sure 'Amount' ($1000) doesn't overpower 'V1' (0.5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into Train (80%) and Test (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=42
        )
        
        # Convert to PyTorch Tensors (The format the GPU/CPU understands)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # 2. Initialize Model
        input_dim = X_train.shape[1] # Automatically detect number of columns (29)
        model = FraudDetector(input_dim)
        
        # 3. Define Training Rules
        criterion = nn.BCELoss() # Binary Cross Entropy Loss (Standard for Fraud detection)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Log Parameters to MLflow (For Auditability)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("input_features", input_dim)

        # 4. Training Loop
        print("Training in progress...")
        for epoch in range(EPOCHS):
            # Forward Pass (Prediction)
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward Pass (Correction)
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Calculate new gradients
            optimizer.step()      # Update weights
            
            if (epoch+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

        # 5. Evaluation (Testing the Model)
        print("Evaluating model...")
        with torch.no_grad(): # Turn off gradient calculation for speed
            y_pred_prob = model(X_test_tensor)
            y_pred_class = y_pred_prob.round() # Convert 0.8 -> 1, 0.2 -> 0
            
            # Calculate Metrics
            # We convert tensors back to numpy for sklearn
            y_test_np = y_test
            y_pred_np = y_pred_class.numpy()
            
            acc = accuracy_score(y_test_np, y_pred_np)
            f1 = f1_score(y_test_np, y_pred_np)
            
            print(f"Results -> Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
            
            # Log Metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("final_loss", loss.item())

        # 6. Save the Model
        # Save locally (Standard PyTorch format)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved locally to: {MODEL_SAVE_PATH}")
        
        # Save to MLflow (MLOps format - allows easy deployment later)
        mlflow.pytorch.log_model(model, "model")
        print("Model tracked in MLflow.")

if __name__ == "__main__":
    train_pipeline()