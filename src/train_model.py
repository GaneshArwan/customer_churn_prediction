import os
import joblib
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_and_preprocess_data

def train_model(X_train, y_train):
    print("Training Logistic Regression (Balanced) model...")
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    model = train_model(X_train, y_train)
    
    model_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
