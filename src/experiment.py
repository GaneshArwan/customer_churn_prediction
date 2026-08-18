import os
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from data_preprocessing import load_and_preprocess_data

def run_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Calculate scale_pos_weight for XGBoost
    # count(negative examples) / count(positive examples)
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    models = {
        "Random Forest (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Logistic Regression (Balanced)": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "XGBoost (Default)": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "XGBoost (Weighted)": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')
    }
    
    print("\n--- Running Experiments ---")
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred)
        print(f"F1-score (Churn=1): {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            
    print("\n--- Testing SMOTE with XGBoost ---")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    smote_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    smote_model.fit(X_train_sm, y_train_sm)
    y_pred_sm = smote_model.predict(X_test)
    f1_sm = f1_score(y_test, y_pred_sm)
    print(f"XGBoost + SMOTE F1-score: {f1_sm:.4f}")
    print(classification_report(y_test, y_pred_sm))
    
    if f1_sm > best_f1:
        best_f1 = f1_sm
        best_model_name = "XGBoost with SMOTE"
        
    print(f"\nBest Model: {best_model_name} with F1-score: {best_f1:.4f}")

if __name__ == "__main__":
    run_experiments()
