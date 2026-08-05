"""Download and extract the SIDDA datasets from Zenodo record 15215272.

Usage:
    python download_data.py --dataset shapes
    python download_data.py --all
"""

import argparse
import os
import urllib.request
import zipfile

from tqdm import tqdm

ZENODO_RECORD = "15215272"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

DATASETS = {
    "shapes": "shapes_dataset.zip",
    "astro_objects": "astronomical_objects_dataset.zip",
    "mnist_m": "mnistm_dataset.zip",
    "gz_evo": "galaxy_dataset.zip",
    "mrssc2": "MRSSC2_dataset.zip",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_file(url: str, dest_path: str) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_dataset(name: str, keep_zip: bool = False) -> str:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choices: {sorted(DATASETS)}")

    file_name = DATASETS[name]
    dest_dir = os.path.join(DATA_DIR, name)

    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        print(f"[{name}] already extracted at {dest_dir}, skipping.")
        return dest_dir

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(zip_path):
        url = f"{ZENODO_BASE}/{file_name}?download=1"
        print(f"[{name}] downloading {url} -> {zip_path}")
        download_file(url, zip_path)
    else:
        print(f"[{name}] zip already present at {zip_path}, skipping download.")

    print(f"[{name}] extracting {zip_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    extract_zip(zip_path, dest_dir)

    if not keep_zip:
        os.remove(zip_path)

    return dest_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SIDDA datasets from Zenodo")
    parser.add_argument(
        "--dataset", type=str, choices=sorted(DATASETS), help="Single dataset to download"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--keep-zip", action="store_true", help="Keep the downloaded zip after extraction"
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Specify --dataset <name> or --all")

    names = sorted(DATASETS) if args.all else [args.dataset]
    for name in names:
        download_dataset(name, keep_zip=args.keep_zip)
