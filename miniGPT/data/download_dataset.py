import urllib.request
from pathlib import Path


URL = (
    "https://www.gutenberg.org/files/1342/1342-0.txt"
)

OUTPUT_PATH = Path(
    "data/raw/dataset.txt"
)


def download_dataset():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading dataset...")

    urllib.request.urlretrieve(
        URL,
        OUTPUT_PATH
    )

    print(
        f"Dataset saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":

    download_dataset()