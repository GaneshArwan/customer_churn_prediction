import html
import os

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.inference import (
    load_and_prepare_preprocessor,
    load_model_json,
    preprocess_input,
)
from src.schemas import CustomerProfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_customer_churn.csv")
MODEL_JSON_PATH = os.path.join(BASE_DIR, "models", "lr_model.json")


@st.cache_data
def _cached_preprocessor(data_path: str):
    return load_and_prepare_preprocessor(data_path)


@st.cache_resource
def _cached_model(model_path: str):
    return load_model_json(model_path)


def classify_risk(prob_churn: float) -> tuple[str, str]:
    """Return CSS class and human-readable risk label for a churn probability."""
    if prob_churn > 0.6:
        return "risk-high", "HIGH RISK"
    if prob_churn > 0.4:
        return "risk-medium", "MEDIUM RISK"
    return "risk-low", "LOW RISK"


def main():
    """Render the Streamlit customer churn prediction application."""
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .metric-card {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            }
            .risk-high { color: #dc3545; font-weight: bold; }
            .risk-medium { color: #fd7e14; font-weight: bold; }
            .risk-low { color: #28a745; font-weight: bold; }

            [data-theme="dark"] .metric-card {
                background-color: #1e1e1e;
                border-color: #333;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        scaler, cat_cols, num_cols, expected_cols, template_X = _cached_preprocessor(
            DATA_PATH
        )
    except FileNotFoundError:
        st.error(
            "Required dataset is unavailable. Run `python src/fetch_data.py` on the server."
        )
        st.stop()
        return

    try:
        model = _cached_model(MODEL_JSON_PATH)
    except FileNotFoundError:
        st.error(
            "Trained model artifact is unavailable. Run `python src/train_model.py` on the server."
        )
        st.stop()
        return

    st.title("📉 Customer Churn Prediction")
    st.markdown(
        "Identify at-risk customers using the class-weighted Logistic Regression model."
    )

    tab1, tab2, tab3 = st.tabs(
        ["🔮 Predict Individual Churn", "📊 Model Comparison", "📈 Data Explorer"]
    )

    with tab1:
        st.markdown("### Customer Profile Input")

        with st.sidebar:
            st.header("Customer Details")
            st.write("Adjust the parameters below to predict churn probability.")

            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

            st.subheader("Account Information")
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
            total_charges = float(tenure * monthly_charges)

            st.subheader("Services")
            phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
            multiplelines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            onlinesecurity = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            onlinebackup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            deviceprotection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            techsupport = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streamingtv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streamingmovies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        raw_input = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phoneservice,
            "MultipleLines": multiplelines,
            "InternetService": internet,
            "OnlineSecurity": onlinesecurity,
            "OnlineBackup": onlinebackup,
            "DeviceProtection": deviceprotection,
            "TechSupport": techsupport,
            "StreamingTV": streamingtv,
            "StreamingMovies": streamingmovies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            validated_input = CustomerProfile(**raw_input).model_dump()
        except ValidationError:
            st.error("Invalid customer profile input provided.")
            st.stop()
            return

        input_df = pd.DataFrame([validated_input])
        X_processed = preprocess_input(
            input_df, scaler, cat_cols, num_cols, expected_cols, template_X
        )

        prob_churn = float(model.predict_proba(X_processed)[0][1])

        st.markdown("### Risk Assessment")
        col1, col2 = st.columns([1, 2])

        with col1:
            risk_class, risk_label = classify_risk(prob_churn)
            safe_risk_class = html.escape(risk_class, quote=True)
            safe_risk_label = html.escape(risk_label)
            safe_prob = html.escape(f"{prob_churn:.1%}")

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Churn Probability</h4>
                    <h1 class="{safe_risk_class}">{safe_prob}</h1>
                    <p>Assessment: <b>{safe_risk_label}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### Recommended Retention Actions")
            if prob_churn > 0.6:
                st.error(
                    "**Immediate Action Required:** High likelihood of cancellation. Consider deploying a targeted retention discount or proactive outreach call."
                )
            elif prob_churn > 0.4:
                st.warning(
                    "**Watch List:** Moderate churn risk. Enroll this account in standard engagement campaigns and monitor service usage."
                )
            else:
                st.success(
                    "**Stable:** Low churn risk. Focus on account expansion and service upgrades rather than retention discounts."
                )

            if hasattr(model, "coef_"):
                coefs = pd.Series(model.coef_[0], index=expected_cols)
                user_impact = coefs * X_processed.iloc[0]
                top_factors = user_impact.sort_values(ascending=False).head(3)
                st.write("**Top Factors Increasing Risk for this Customer:**")
                for feature, impact in top_factors.items():
                    if impact > 0:
                        st.write(f"- 🔴 **{feature}**")

    with tab2:
        st.markdown("### 📊 Model Comparison & Architecture")
        st.write(
            "This dashboard uses the **Class-Weighted Logistic Regression** model selected over the Baseline Random Forest to prioritize recall on churning accounts."
        )
        st.markdown(
            """
            | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
            |---|---|---|---|---|---|---|---|
            | Random Forest (Baseline) | 78.5% | 0.63 | 47.3% | 0.54 | 0.815 | 0.605 | 0.148 |
            | Logistic Regression (Balanced) | 73.1% | 0.50 | 79.1% | 0.61 | 0.832 | 0.628 | 0.173 |
            | Deep Learning (MLP) | 78.0% | 0.63 | 43.3% | 0.51 | 0.829 | 0.637 | 0.143 |
            """
        )

        diag_plots = [
            ("roc_curve.png", "ROC Curve Comparison"),
            ("pr_curve.png", "Precision-Recall Curve Comparison"),
            ("lr_coef.png", "Top 10 LR Log-Odds Coefficients"),
        ]
        available_diag = [
            (os.path.join(BASE_DIR, "plots", fname), caption)
            for fname, caption in diag_plots
            if os.path.exists(os.path.join(BASE_DIR, "plots", fname))
        ]
        if available_diag:
            col_d1, col_d2 = st.columns(2)
            for idx, (img_path, caption) in enumerate(available_diag):
                target_col = col_d1 if idx % 2 == 0 else col_d2
                with target_col:
                    st.image(img_path, caption=caption)
        else:
            st.write(
                "Model comparison plots not found. Run `python src/generate_report.py` first."
            )

    with tab3:
        st.markdown("### 📈 Data Explorer")
        st.write("Historical exploratory data analysis charts.")

        plots_dir = os.path.join(BASE_DIR, "plots")
        plot_files = [
            "churn_distribution.png",
            "monthly_charges_vs_churn.png",
            "contract_vs_churn.png",
            "tenure_vs_churn.png",
        ]
        if any(os.path.exists(os.path.join(plots_dir, name)) for name in plot_files):
            col_a, col_b = st.columns(2)
            with col_a:
                for name in plot_files[:2]:
                    path = os.path.join(plots_dir, name)
                    if os.path.exists(path):
                        st.image(path)
            with col_b:
                for name in plot_files[2:]:
                    path = os.path.join(plots_dir, name)
                    if os.path.exists(path):
                        st.image(path)
        else:
            st.write("EDA plots not found. Run `python src/eda.py` first.")


if __name__ == "__main__":
    main()
