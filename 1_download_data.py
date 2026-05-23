"""
Step 1: Download the Home Credit Default Risk dataset from Kaggle.

Prerequisites:
  pip install kaggle
  Place kaggle.json in ~/.kaggle/kaggle.json  (download from Kaggle → Account → API)
"""

import os
import subprocess
import zipfile

DEST = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DEST, exist_ok=True)

def download():
    print("Downloading Home Credit Default Risk dataset...")
    subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", "home-credit-default-risk", "-p", DEST],
        check=True,
    )

    zip_path = os.path.join(DEST, "home-credit-default-risk.zip")
    if os.path.exists(zip_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            # Only extract the main training file to keep things fast
            for name in z.namelist():
                if name in ("application_train.csv", "application_test.csv"):
                    z.extract(name, DEST)
        os.remove(zip_path)
        print(f"Done. Files saved to: {DEST}/")
    else:
        print("Download may have failed. Check your Kaggle API credentials.")

if __name__ == "__main__":
    download()
