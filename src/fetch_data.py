import urllib.request
import os

def main():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dest_path = os.path.join(dest_dir, "telco_customer_churn.csv")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)  # nosec B310
        print(f"Dataset downloaded to {dest_path}")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    main()
