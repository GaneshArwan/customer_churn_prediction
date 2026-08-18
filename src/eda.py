import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Generating EDA plots...")
    
    # 1. Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Customer Churn Distribution')
    plt.savefig(os.path.join(plots_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
    plt.title('Tenure Distribution by Churn')
    plt.savefig(os.path.join(plots_dir, 'tenure_vs_churn.png'))
    plt.close()
    
    # 3. MonthlyCharges vs Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title('Monthly Charges by Churn')
    plt.savefig(os.path.join(plots_dir, 'monthly_charges_vs_churn.png'))
    plt.close()
    
    # 4. Contract type vs Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Contract Type vs Churn')
    plt.savefig(os.path.join(plots_dir, 'contract_vs_churn.png'))
    plt.close()
    
    print(f"EDA plots saved to {plots_dir}")

if __name__ == "__main__":
    main()
