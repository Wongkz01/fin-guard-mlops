import pandas as pd
import sqlite3
import os
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# --- CONFIGURATION ---
DB_PATH = os.path.join("data", "fraud_data.db")
REPORT_PATH = "drift_report.html"

def load_data():
    """Load data from SQL"""
    print("Loading data from SQL...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df

def simulate_drift(df):
    """
    Splits data into 'Reference' (Old) and 'Current' (New).
    We artificially BREAK the new data to show you what an alert looks like.
    """
    # Split: First 5000 rows are "Training Data", Last 5000 are "Production Data"
    reference_data = df.iloc[:5000].copy()
    current_data = df.iloc[5000:].copy()

    # --- SIMULATE FRAUD PATTERN CHANGE ---
    print("Simulating Data Drift...")
    
    # 1. Scammers start stealing HUGE amounts (Drift in 'Amount')
    # We multiply amounts by 10 to make them look suspicious
    current_data['Amount'] = current_data['Amount'] * 10 + 500
    
    # 2. Scammers change their behavior slightly (Drift in 'V1')
    current_data['V1'] = current_data['V1'] + 5.0
    
    return reference_data, current_data

def generate_report(reference, current):
    """Generates an HTML report showing the drift"""
    print("Calculating Drift Metrics...")
    
    # Create a report looking for Data Drift
    drift_report = Report([
        DataDriftPreset(), 
    ])
    
    # Run the calculation
    my_eval = drift_report.run(reference_data=reference, current_data=current)

    # Save the report
    my_eval.save_html(REPORT_PATH)
    print(f"✅ Report saved to {REPORT_PATH}")
    print("Open this file in your browser to see the dashboard!")

if __name__ == "__main__":
    # 1. Load
    full_data = load_data()
    
    # 2. Simulate "New Data" appearing
    ref, curr = simulate_drift(full_data)
    
    # 3. Check for issues
    generate_report(ref, curr)