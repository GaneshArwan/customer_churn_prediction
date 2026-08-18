import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data

def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    model_path = os.path.join(base_dir, "models", "rf_model.pkl")
    plots_dir = os.path.join(base_dir, "plots")
    
    print("Loading data and model...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    model = joblib.load(model_path)
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix plot saved to {plots_dir}")

if __name__ == "__main__":
    evaluate_model()
