import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Drop customerID as it's not a useful feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # TotalCharges is object (string), convert to numeric and handle errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with median or drop them
    # There are only 11 missing values typically in this dataset
    df = df.dropna(subset=['TotalCharges'])
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y) # Yes -> 1, No -> 0
    
    # Get categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical variables
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    print(f"Data preprocessed successfully.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
