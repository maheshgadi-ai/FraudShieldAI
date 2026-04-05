# FraudShieldAI

A production-grade, end-to-end fraud detection system combining supervised learning, deep learning, and a hybrid ensemble — built as a capstone ML portfolio project.

---

## Overview

FraudShieldAI detects fraudulent credit card transactions using a multi-model approach:

| Model Family | Algorithms |
|---|---|
| Supervised Learning | Logistic Regression, Random Forest, XGBoost |
| Deep Learning | Feedforward Neural Network (FNN), LSTM |
| Ensemble | Soft-voting / Stacking hybrid |

The system handles the core challenges of real-world fraud detection: severe class imbalance (~0.5% fraud rate), high-volume data (1.3M transactions), and the need for interpretable predictions via SHAP.

---

## Architecture

```
FraudShieldAI/
├── data/
│   ├── raw/                  # Original Kaggle CSVs (gitignored)
│   └── processed/            # Engineered features, parquet format
├── notebooks/                # EDA and prototyping notebooks
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Cleaning, encoding, scaling
│   │   └── features.py       # Feature engineering (velocity, geo, time)
│   ├── models/
│   │   ├── trainer.py        # Orchestrates all model training
│   │   ├── supervised.py     # LR, RF, XGBoost
│   │   ├── fnn.py            # PyTorch FNN
│   │   ├── lstm.py           # PyTorch LSTM
│   │   └── ensemble.py       # Hybrid ensemble
│   ├── evaluation/
│   │   ├── evaluator.py      # Metrics, plots, SHAP
│   │   └── metrics.py        # Precision, Recall, F1, AUC-ROC
│   └── utils/
│       ├── config.py         # Config loader
│       └── helpers.py        # Shared utilities
├── app/
│   └── streamlit_app.py      # Web interface
├── configs/
│   ├── pipeline.yaml         # Data paths, preprocessing settings
│   └── models.yaml           # Hyperparameters for all models
├── tests/                    # Unit tests
├── outputs/                  # Generated models, metrics, plots (gitignored)
├── main.py                   # Pipeline entrypoint
└── requirements.txt
```

---

## Dataset

**Source:** [Credit Card Transactions Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

- 1.3 million transactions
- Features: transaction amount, merchant info, cardholder location, timestamp
- Target: `is_fraud` (binary, ~0.5% positive rate)

### Download

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/FraudShieldAI.git
cd FraudShieldAI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

```bash
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

### 3. Run the full pipeline

```bash
# All stages: preprocess → train → evaluate
python main.py --stage all

# Or run individual stages
python main.py --stage preprocess
python main.py --stage train
python main.py --stage evaluate
```

### 4. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| `geo_distance` | Haversine distance between cardholder and merchant |
| `tx_velocity_1d` | Number of transactions by card in past 24h |
| `tx_velocity_7d` | Number of transactions by card in past 7 days |
| `spend_rolling_mean` | Rolling mean spend per card (30-day window) |
| `spend_rolling_std` | Rolling std dev of spend per card |
| `hour` | Hour of transaction |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary weekend indicator |
| `age` | Cardholder age derived from DOB |

---

## Model Performance

> Results populated after running the pipeline. See `outputs/metrics/` for full reports.

| Model | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| FNN | — | — | — | — |
| LSTM | — | — | — | — |
| Ensemble | — | — | — | — |

---

## Explainability

SHAP (SHapley Additive exPlanations) values are computed for tree-based and neural models to explain individual predictions. Outputs saved to `outputs/shap/`.

---

## Web Application

The Streamlit app supports:
- Upload a CSV of transactions
- Real-time fraud predictions with confidence scores
- Feature importance and SHAP visualizations
- Fraud trend dashboard

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Technologies

- **Python 3.10+**
- **Scikit-learn** — supervised learning, preprocessing, SMOTE
- **XGBoost** — gradient boosted trees
- **PyTorch** — FNN and LSTM models
- **SHAP** — model explainability
- **Optuna** — hyperparameter optimization
- **Streamlit** — web application
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn / Plotly** — visualization

---

## License

MIT
