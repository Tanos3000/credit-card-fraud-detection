"""Downloads the ULB credit card fraud dataset into data/creditcard.csv.

Original source: Kaggle (mlg-ulb/creditcardfraud), which requires a login to
download directly. This script instead uses a Zenodo mirror of the same file
(CC-BY-4.0, DOI 10.5281/zenodo.7395559) so the dataset can be fetched with a
plain HTTP request, no account needed.
"""

import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi

URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
DATA_DIR = Path(__file__).parent / "data"
TARGET = DATA_DIR / "creditcard.csv"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return

    print("Downloading dataset from Zenodo (150 MB, may take a moment)...")
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(URL, context=context) as response:
        TARGET.write_bytes(response.read())

    print(f"Saved to {TARGET}")


if __name__ == "__main__":
    main()
