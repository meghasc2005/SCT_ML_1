import os
import urllib.request
import ssl

MIRROR_URLS = [
    "https://raw.githubusercontent.com/Shitao-zz/Kaggle-House-Prices-Advanced-Regression-Techniques/master/input/train.csv",
    "https://raw.githubusercontent.com/data-doctors/kaggle-house-prices-advanced-regression-techniques/master/data/train.csv",
    "https://raw.githubusercontent.com/SunnyMarkLiu/Kaggle-House-Prices/master/data/train.csv"
]

def download_train_data(dest_path="data/train.csv"):
    """
    Downloads Kaggle 'House Prices - Advanced Regression Techniques' train.csv
    to dest_path if not already present.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        print(f"Dataset already exists at {dest_path} ({os.path.getsize(dest_path)} bytes).")
        return dest_path

    print(f"Downloading Kaggle House Prices train.csv to {dest_path}...")
    ctx = ssl._create_unverified_context()
    
    for url in MIRROR_URLS:
        try:
            print(f"Attempting download from mirror: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
                content = response.read()
                # Verify header contains key columns
                header = content.decode('utf-8', errors='ignore').split('\n')[0]
                if "GrLivArea" in header and "SalePrice" in header:
                    with open(dest_path, "wb") as f:
                        f.write(content)
                    print(f"Successfully downloaded dataset ({len(content)} bytes) to {dest_path}.")
                    return dest_path
                else:
                    print(f"Mirror downloaded content missing required headers.")
        except Exception as e:
            print(f"Mirror failed ({url}): {e}")
            
    raise RuntimeError("Failed to download train.csv from all mirrors. Please place train.csv manually in data/train.csv.")

if __name__ == "__main__":
    download_train_data()
