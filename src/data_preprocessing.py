import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(file_path):
    """Load Telco churn CSV, clean TotalCharges, encode categories, and scale without leakage."""
    df = pd.read_csv(file_path)

    # Drop customerID as it is an identifier rather than a predictive feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # TotalCharges is stored as strings with whitespace for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    le = LabelEncoder()
    y = le.fit_transform(y)  # Yes -> 1, No -> 0

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Split before fitting StandardScaler to prevent test-set statistics from leaking into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    print("Data preprocessed successfully.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
