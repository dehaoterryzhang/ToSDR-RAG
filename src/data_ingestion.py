import kagglehub
import shutil
from pathlib import Path

def download_tosdr_dataset():
    """
    Downloads the ToS;DR dataset using kagglehub and stores it under data/raw/.
    If the dataset already exists, it skips downloading to ensure reproducibility.
    """
    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists (any .csv or .json files)
    existing_files = list(target_dir.glob("*"))
    if any(f.suffix in [".csv", ".json"] for f in existing_files):
        print(f"✅ Dataset already exists in {target_dir.resolve()}, skipping download.")
        return target_dir

    print("📦 Downloading ToS;DR dataset from KaggleHub...")
    dataset_path = Path(kagglehub.dataset_download("sonu1607/tosdr-terms-of-service-corpus"))

    # Copy all files (recursively) to data/raw
    for item in dataset_path.iterdir():
        dest = target_dir / item.name
        if item.is_file():
            shutil.copy(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)

    print(f"✅ Dataset successfully stored in: {target_dir.resolve()}")
    return target_dir

if __name__ == "__main__":
    download_tosdr_dataset()
