# Customer Churn Prediction

![CI Pipeline](https://github.com/GaneshArwan/customer_churn_prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-66%25-brightgreen)
![Tests](https://img.shields.io/badge/Tests-61%20passed-brightgreen)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GaneshArwan/customer_churn_prediction/blob/main/notebooks/churn_prediction_walkthrough.ipynb)
![License](https://img.shields.io/badge/License-MIT-green)

An unweighted Random Forest catches **47.3%** of churning customers on the IBM Telco dataset. A class-weighted Logistic Regression catches **79.1%** (+31.8 percentage points) on the same 1,407-customer holdout split. This repository contains the preprocessing pipeline, three-model benchmark, Streamlit prediction dashboard, Google Colab walkthrough, and an IEEE-formatted PDF report.

## Decision Problem
Subscription telecom teams pay far more to acquire a subscriber than to keep one. Standard classifiers optimized for overall accuracy under-predict churn because only **26.6%** of accounts in the dataset cancel. The objective is to rank churn risk reliably, raise minority-class recall, and quantify net retention profit after campaign outreach costs.

## Dataset Audit
- **Source:** [Telco Customer Churn (Kaggle / IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** `7,043` raw rows × `21` columns; `11` rows with blank `TotalCharges` (`tenure = 0`) are dropped, leaving `7,032` usable accounts (`26.6%` churn, `73.4%` retained).
- **Split:** 80/20 (`random_state=42`) into `5,625` training rows and `1,407` holdout test rows (`374` churners). `StandardScaler` is fit on the training split only.

## Holdout Results (`n = 1,407`)
| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Random Forest (Baseline) | 78.5% | 0.63 | 47.3% | 0.54 | 0.815 | 0.605 | 0.148 |
| Logistic Regression (Balanced) | 73.1% | 0.50 | 79.1% | 0.61 | 0.832 | 0.628 | 0.173 |
| Deep Learning (MLP, Early Stopping) | 78.0% | 0.63 | 43.3% | 0.51 | 0.829 | 0.637 | 0.143 |

- **Cross-Validation Stability (5-Fold Stratified on Training Split):** Balanced Logistic Regression achieves `0.847 ± 0.013` ROC-AUC and `0.800 ± 0.032` churn recall, compared to `0.829 ± 0.010` ROC-AUC and `0.500 ± 0.023` recall for the Random Forest baseline.
- **Retention Campaign Economics ($780 LTV, $50 Outreach Cost):** On `1,407` evaluated accounts, Balanced Logistic Regression identifies `296` true churners (`300` false positives) versus `177` (`106` false positives) for Random Forest. Net profit increases by **+$12,196** at a 30% save rate (breakeven save rate: `12.9%`) and **+$30,760** at a 50% save rate.
- **Calibration Caveat:** Class reweighting shifts mean predicted probability upward (`0.416` vs. `0.266` prevalence), raising Brier score (`0.173` vs. `0.143` for MLP). Fit Platt scaling or isotonic calibration on a validation fold before thresholding expected-value spend.

## Run

### Option 1: Interactive Dashboard (Local or Streamlit Cloud)
`app.py` validates inputs with Pydantic v2 (`CustomerProfile`) and loads model weights from JSON (`models/lr_model.json`) without `pickle` or `joblib.load`. If `data/` or `models/` are absent on a fresh clone or cloud container, `app.py` downloads the dataset over HTTPS and trains `models/lr_model.json` on startup.

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Full Pipeline & IEEE PDF Report
```bash
python src/fetch_data.py           # pulls Telco CSV over HTTPS -> data/
python src/eda.py                  # EDA charts -> plots/
python src/train_model.py          # trains Balanced LR -> models/lr_model.json + models/rf_model.pkl
python src/evaluate.py             # test metrics + confusion matrix -> plots/confusion_matrix.png
python src/generate_report.py      # 3-model CV + ROI benchmark -> churn_prediction_report.pdf
```

### Option 3: Docker
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

### Option 4: Google Colab
Open [notebooks/churn_prediction_walkthrough.ipynb](notebooks/churn_prediction_walkthrough.ipynb) via the **Open in Colab** badge above to run preprocessing, 5-fold CV, ROC/PR/coefficient plots, and ROI math in browser with zero local setup.

## Tests & CI
```bash
pytest --cov=src --cov=app --cov-report=term-missing -v
```
**62 tests** pass at **66%** total coverage (`40%` minimum gate). GitHub Actions runs a single cached workflow (`.github/workflows/ci.yml`) restricted to code/dependency path changes (`pytest`, `bandit -r src/ app.py`, `pip-audit -r requirements.txt`).

| Module | Coverage |
|---|---|
| `src/inference.py` | 98% |
| `src/schemas.py` | 100% |
| `src/eda.py` | 97% |
| `src/evaluate.py` | 97% |
| `src/train_model.py` | 96% |
| `src/fetch_data.py` | 92% |
| `app.py` | 78% |
| `src/data_preprocessing.py` | 80% |
| `src/generate_report.py` | 47% |
| `src/experiment.py` | 0% |
| **Total** | **66%** |

## Repository Layout
- `app.py` — Streamlit dashboard with Pydantic validation, HTML escaping, and startup artifact bootstrap
- `src/schemas.py` — `CustomerProfile` input validation schema (`extra="forbid"`, finite bounds)
- `src/inference.py` — Preprocessing alignment, JSON model serialization, and startup bootstrap
- `src/data_preprocessing.py` — Data cleaning, 80/20 split, and leak-free `StandardScaler`
- `src/fetch_data.py` — HTTPS dataset download with timeout and redirect-scheme checks
- `src/eda.py` — Distribution and tenure/contract EDA charts
- `src/train_model.py` — Balanced Logistic Regression training script
- `src/evaluate.py` — Holdout evaluation and confusion matrix generation
- `src/experiment.py` — 6-variant benchmark including XGBoost and SMOTE
- `src/generate_report.py` — Builds `churn_prediction_report.pdf` (IEEE two-column paper)
- `notebooks/churn_prediction_walkthrough.ipynb` — Standalone Colab notebook
- `GUARDRAILS.md` — Input, ML pipeline, container, and security guardrails
- `tests/` — Unit and regression test suite

## Limitations
- Predictions rely on cross-sectional snapshot features without macroeconomic indicators or prior outreach history.
- The MLP tuning budget used 5 random configurations without explicit class weighting; focal loss or threshold tuning could improve its minority-class recall.
