# FinGuard: MLOps Fraud Detection System

**FinGuard** is an end-to-end MLOps system designed to detect fraudulent financial transactions. It demonstrates a complete production lifecycle, from data ingestion to real-time inference, containerization, and automated drift monitoring.

## 🚀 Key Features (Aligned with AI Engineering Standards)

* **Production-Grade Training:** PyTorch neural network wrapped in **MLflow** for experiment tracking and reproducibility.
* **Robust Data Pipeline:** SQL-based ingestion simulating real-world batch processing.
* **Microservice Deployment:** Real-time inference API built with **FastAPI**, serving predictions under 50ms.
* **Infrastructure as Code:** Fully Dockerized application ensuring environment consistency across Windows/Linux.
* **CI/CD Automation:** **GitHub Actions** pipeline that automatically runs unit tests on every commit to ensure system stability.
* **AI Governance & Monitoring:** Automated **Evidently AI** pipeline to detect data drift (e.g., shifts in fraud patterns) and ensure model reliability.

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Modeling:** PyTorch, Scikit-Learn
* **Ops & Tracking:** MLflow
* **Serving:** FastAPI, Uvicorn
* **Infrastructure:** Docker
* **Monitoring:** Evidently AI
* **Database:** SQLite

## 📂 Project Structure

```text
fin-guard-mlops/
├── .github/workflows/   # CI/CD (GitHub Actions)
├── data/                # SQL Data Storage
├── models/              # Trained PyTorch artifacts
├── src/
│   ├── api/             # FastAPI Inference Service
│   ├── data_pipeline/   # SQL Ingestion Scripts
│   ├── training/        # Model Training & MLflow Logging
│   └── monitoring/      # Data Drift Detection
├── tests/               # Pytest Unit Tests
└── Dockerfile           # Container Configuration