import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime

# CONFIGURATION
DB_PATH = os.path.join("data", "fraud_data.db")
TABLE_NAME = "transactions"

def generate_synthetic_data(n_rows=10000):
    """
    Generates synthetic transaction data mimicking PCA-transformed credit card data.
    """
    print(f"Generating {n_rows} rows of synthetic data...")
    
    np.random.seed(42) # For reproducibility
    
    # Simulate 'Time' (seconds elapsed)
    time_val = np.arange(0, n_rows)
    
    # Simulate V1-V28 (Anonymized features usually found in banking data)
    # We create random normal distributions
    features = np.random.randn(n_rows, 28)
    feature_cols = [f'V{i+1}' for i in range(28)]
    
    # Simulate 'Amount'
    amounts = np.random.exponential(scale=100, size=n_rows)
    
    # Simulate 'Class' (0 = Legitimate, 1 = Fraud)
    # 0.5% fraud rate
    classes = np.random.choice([0, 1], size=n_rows, p=[0.995, 0.005])
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_cols)
    df['Time'] = time_val
    df['Amount'] = amounts
    df['Class'] = classes
    
    return df

def load_to_db(df):
    """
    Loads the dataframe into a SQLite database.
    """
    print(f"Loading data into {DB_PATH}...")
    
    # Connect to SQLite (creates file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    
    # Write data to SQL
    # if_exists='replace' mimics a fresh ingestion batch
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    
    conn.close()
    print("Data ingestion complete.")

def verify_data():
    """
    Simple SQL query to verify data integrity.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # SQL Query to check count and fraud distribution
    cursor.execute(f"SELECT Class, COUNT(*) FROM {TABLE_NAME} GROUP BY Class")
    results = cursor.fetchall()
    
    print("\n--- Data Verification (SQL) ---")
    print(f"Checking table: {TABLE_NAME}")
    for row in results:
        label = "Fraud" if row[0] == 1 else "Legitimate"
        print(f"Class {row[0]} ({label}): {row[1]} transactions")
        
    conn.close()

if __name__ == "__main__":
    # 1. Generate
    data = generate_synthetic_data()
    # 2. Load
    load_to_db(data)
    # 3. Verify
    verify_data()