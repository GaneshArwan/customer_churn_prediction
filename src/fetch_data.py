import os
import urllib.parse
import urllib.request


def main():
    """Download the Telco Customer Churn dataset over HTTPS with timeout and redirect checks."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Refusing non-HTTPS dataset URL scheme: {parsed.scheme}")

    dest_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    dest_path = os.path.join(dest_dir, "telco_customer_churn.csv")

    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading dataset from {url}...")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            final_scheme = urllib.parse.urlparse(resp.geturl()).scheme
            if final_scheme != "https":
                raise ValueError(f"Refusing non-HTTPS redirect target: {final_scheme}")
            payload = resp.read()
        with open(dest_path, "wb") as out_file:
            out_file.write(payload)
        print(f"Dataset downloaded to {dest_path}")
    except Exception as e:
        print(f"Failed to download: {e}")


if __name__ == "__main__":
    main()
