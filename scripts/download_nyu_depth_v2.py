"""
Download NYU Depth V2 dataset for MCResNet training.

This script downloads the official NYU Depth V2 labeled dataset
and saves it to a specified location.
"""

import urllib.request
import os
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_nyu_depth_v2(save_dir: str = "./data", filename: str = "nyu_depth_v2_labeled.mat"):
    """
    Download NYU Depth V2 dataset.

    Args:
        save_dir: Directory to save the dataset (default: ./data)
        filename: Filename for the dataset (default: nyu_depth_v2_labeled.mat)

    Returns:
        Path to the downloaded file
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / filename

    # Check if file already exists
    if file_path.exists():
        print(f"✅ Dataset already exists at: {file_path}")
        print(f"   Size: {file_path.stat().st_size / (1024**3):.2f} GB")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Using existing file.")
            return str(file_path)

    # Download URL
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    print("=" * 60)
    print("NYU Depth V2 Dataset Download")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Destination: {file_path}")
    print(f"Expected size: ~2.8 GB")
    print("=" * 60)
    print("\nDownloading...")

    try:
        # Download with progress bar
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)

        print(f"\n✅ Download complete!")
        print(f"   Saved to: {file_path}")
        print(f"   Size: {file_path.stat().st_size / (1024**3):.2f} GB")

        return str(file_path)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if file_path.exists():
            print(f"Removing incomplete file...")
            file_path.unlink()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NYU Depth V2 dataset")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data",
        help="Directory to save the dataset (default: ./data)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="nyu_depth_v2_labeled.mat",
        help="Filename for the dataset (default: nyu_depth_v2_labeled.mat)"
    )

    args = parser.parse_args()

    # Download
    dataset_path = download_nyu_depth_v2(args.save_dir, args.filename)

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(f"1. Use this path in your training script:")
    print(f"   NYU_DATASET_PATH = '{dataset_path}'")
    print(f"\n2. Load the dataset:")
    print(f"   from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders")
    print(f"   train_loader, val_loader = create_nyu_dataloaders('{dataset_path}')")
    print("\n3. Train your model!")
    print("=" * 60)
