# Project Guardrails

This document defines the runtime, machine learning, and security guardrails enforced in `customer_churn_prediction`.

## 1. Input Validation Guardrails
- **Strict Pydantic v2 Schema (`src/schemas.py`)**: All interactive prediction requests pass through `CustomerProfile` (`extra="forbid"`) before entering preprocessing or model inference.
  - Categorical fields are restricted via `Literal[...]` to exact Telco feature values.
  - `tenure` is bounded to `0..120`, `MonthlyCharges` to `0.0..1000.0` (`allow_inf_nan=False`), and `TotalCharges` to `0.0..120000.0` (`allow_inf_nan=False`).
- **HTML Output Escaping (`app.py`)**: Interpolated card fields in `st.markdown(..., unsafe_allow_html=True)` are escaped with `html.escape` to prevent XSS.
- **Safe Error Banners (`app.py`)**: Missing data/model errors return generic operator instructions without disclosing server filesystem paths.

## 2. ML Pipeline Guardrails
- **Leak-Free Scaling (`src/data_preprocessing.py`, `src/inference.py`)**: `train_test_split(..., test_size=0.2, random_state=42)` executes before `StandardScaler.fit(...)` so test-set statistics never leak into training or inference scaling.
- **Drift-Free One-Hot Alignment (`src/inference.py`)**: `preprocess_input` expands inference rows with `drop_first=False` and aligns columns via `.reindex(columns=expected_cols, fill_value=0)`, avoiding baseline category drift.
- **Recall-Prioritized Class Weighting (`src/train_model.py`)**: `LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)` prevents minority-class churners from being ignored.

## 3. Security & Infrastructure Guardrails
- **Zero-Pickle Runtime Loading (`src/inference.py`, `src/train_model.py`, `src/evaluate.py`, `app.py`)**: Runtime inference and evaluation load model parameters from JSON (`models/lr_model.json`) via `load_model_json`, avoiding `pickle`/`joblib.load` deserialization (`CWE-502`).
- **Static Analysis (SAST)**: CI runs `bandit -r src/ app.py` with explicit `permissions: contents: read` on every push and pull request to `main`.
- **Dependency Vulnerability Scanning**: CI runs `pip-audit -r requirements.txt`.
- **HTTPS-Only Data Ingestion (`src/fetch_data.py`)**: Enforces `https` scheme before and after redirects with a 30-second network timeout.
- **Container Hardening (`Dockerfile`)**: Multi-stage build drops `build-essential` from runtime, keeps `/app` read-only (`root:root`, `a+rX`) for non-root `appuser`, disables Streamlit usage telemetry, and defines a `HEALTHCHECK` probing `/_stcore/health`.
- **No Side Effects on Import (`app.py`)**: Streamlit UI setup and rendering are isolated inside `main()` under `if __name__ == "__main__":`.
