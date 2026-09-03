import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- UI Designer System Config ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI aesthetics (UI Designer persona)
st.markdown("""
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
""", unsafe_allow_html=True)

# --- Path & Model Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_customer_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

@st.cache_data
def load_and_prepare_preprocessor(data_path):
    """Loads dataset and fits the scaler/dummies so we can process single rows identically."""
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please run `python src/fetch_data.py`")
        st.stop()
        
    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    X = df.drop('Churn', axis=1)
    
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X[numerical_cols])
    
    # Get expected dummy columns
    X_dummy = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    expected_cols = X_dummy.columns.tolist()
    
    return scaler, categorical_cols, numerical_cols, expected_cols, X

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please run `python src/train_model.py`")
        st.stop()
    return joblib.load(MODEL_PATH)

def preprocess_input(input_df, scaler, categorical_cols, numerical_cols, expected_cols, template_df):
    """Preprocess a single row of input using the fitted properties."""
    # Concatenate with a template row to ensure all categorical levels exist before getting dummies
    combined = pd.concat([template_df.iloc[[0]], input_df], ignore_index=True)
    
    combined[numerical_cols] = scaler.transform(combined[numerical_cols])
    combined_dummy = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
    
    # Realign columns to match training exactly
    combined_dummy = combined_dummy.reindex(columns=expected_cols, fill_value=0)
    
    # Return just the target row (index 1)
    return combined_dummy.iloc[[1]]

# --- Load Assets ---
scaler, cat_cols, num_cols, expected_cols, template_X = load_and_prepare_preprocessor(DATA_PATH)
model = load_model()

# --- UI Layout ---
st.title("📉 Customer Churn Prediction")
st.markdown("Identify at-risk customers instantly using our class-weighted Logistic Regression model.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Individual Churn", "📊 Model Comparison", "📈 Data Explorer"])

with tab1:
    st.markdown("### Customer Profile Input")
    
    with st.sidebar:
        st.header("Customer Details")
        st.write("Adjust the parameters below to predict churn probability.")
        
        # Demographics
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        # Account
        st.subheader("Account Information")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
        total_charges = tenure * monthly_charges # simplification for UI input
        
        # Services
        st.subheader("Services")
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Build input dataframe
    input_data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phoneservice, 'MultipleLines': multiplelines,
        'InternetService': internet, 'OnlineSecurity': onlinesecurity, 'OnlineBackup': onlinebackup,
        'DeviceProtection': deviceprotection, 'TechSupport': techsupport, 'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies, 'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([input_data])
    
    # Process and predict
    X_processed = preprocess_input(input_df, scaler, cat_cols, num_cols, expected_cols, template_X)
    
    # The model we saved is LogisticRegression (despite being named rf_model.pkl originally)
    prob_churn = model.predict_proba(X_processed)[0][1]
    is_churn = model.predict(X_processed)[0]
    
    # Analytics Reporter Persona: Actionable Insights
    st.markdown("### Risk Assessment")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prob_churn > 0.6:
            risk_class = "risk-high"
            risk_label = "HIGH RISK"
        elif prob_churn > 0.4:
            risk_class = "risk-medium"
            risk_label = "MEDIUM RISK"
        else:
            risk_class = "risk-low"
            risk_label = "LOW RISK"
            
        st.markdown(f"""
        <div class="metric-card">
            <h4>Churn Probability</h4>
            <h1 class="{risk_class}">{prob_churn:.1%}</h1>
            <p>Assessment: <b>{risk_label}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("#### Strategic Recommendations (Analytics Reporter)")
        if prob_churn > 0.6:
            st.error("**Immediate Action Required:** This customer has a very high likelihood of canceling. Consider deploying an immediate retention discount or proactive outreach call.")
        elif prob_churn > 0.4:
            st.warning("**Watch List:** This customer is showing signs of churn. Ensure they are enrolled in standard engagement campaigns and monitor usage drops.")
        else:
            st.success("**Stable:** This customer is currently stable. Focus on upsell opportunities rather than retention discounts.")
            
        # Display underlying feature contributions if it's a linear model
        if hasattr(model, 'coef_'):
            coefs = pd.Series(model.coef_[0], index=expected_cols)
            # Find the top 3 factors increasing churn risk for THIS user (coef * value)
            user_impact = coefs * X_processed.iloc[0]
            top_factors = user_impact.sort_values(ascending=False).head(3)
            st.write("**Top Factors Increasing Risk for this Customer:**")
            for feature, impact in top_factors.items():
                if impact > 0:
                    st.write(f"- 🔴 **{feature}**")
                    
with tab2:
    st.markdown("### 📊 Model Comparison & Architecture")
    st.write("This dashboard is powered by the **Class-Weighted Logistic Regression** model. We selected this over the Baseline Random Forest because it maximizes *Recall*, successfully identifying 79% of true churners (compared to 48% for the baseline).")
    
    try:
        roc_path = os.path.join(BASE_DIR, "plots", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve Comparison")
    except Exception as e:
        st.write("ROC curve image not found. Run the report generator first.")
        
with tab3:
    st.markdown("### 📈 Data Explorer")
    st.write("Historical exploratory data analysis charts.")
    
    try:
        plots_dir = os.path.join(BASE_DIR, "plots")
        colA, colB = st.columns(2)
        with colA:
            if os.path.exists(os.path.join(plots_dir, "churn_distribution.png")):
                st.image(os.path.join(plots_dir, "churn_distribution.png"))
            if os.path.exists(os.path.join(plots_dir, "monthly_charges_vs_churn.png")):
                st.image(os.path.join(plots_dir, "monthly_charges_vs_churn.png"))
        with colB:
            if os.path.exists(os.path.join(plots_dir, "contract_vs_churn.png")):
                st.image(os.path.join(plots_dir, "contract_vs_churn.png"))
            if os.path.exists(os.path.join(plots_dir, "tenure_vs_churn.png")):
                st.image(os.path.join(plots_dir, "tenure_vs_churn.png"))
    except Exception as e:
        st.write("EDA plots not found. Run `python src/eda.py` first.")
