#!/usr/bin/env python3
"""Download script for datasets."""

import argparse
import os
import requests
from pathlib import Path
import tarfile
import zipfile


def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {destination}")


def extract_archive(archive_path, extract_to):
    """Extract archive to specified directory."""
    print(f"Extracting {archive_path}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"Extracted to {extract_to}")


def main():
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data-root', type=str, required=True, help='Data root directory')
    parser.add_argument('--dataset', type=str, choices=['sample_multimodal'], 
                       default='sample_multimodal', help='Dataset to download')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'sample_multimodal':
        # Create sample multimodal data structure
        rgb_light_dir = data_root / 'multimodal' / 'rgb_light_meter'
        rgb_nir_dir = data_root / 'multimodal' / 'rgb_nir'
        
        (rgb_light_dir / 'rgb').mkdir(parents=True, exist_ok=True)
        (rgb_light_dir / 'light_meter').mkdir(parents=True, exist_ok=True)
        (rgb_nir_dir / 'rgb').mkdir(parents=True, exist_ok=True)
        (rgb_nir_dir / 'nir').mkdir(parents=True, exist_ok=True)
        
        # Create placeholder files
        with open(rgb_light_dir / 'README.md', 'w') as f:
            f.write("# RGB + Light Meter Dataset\n\nPlaceholder for multimodal sensor data.")
        
        with open(rgb_nir_dir / 'README.md', 'w') as f:
            f.write("# RGB + NIR Dataset\n\nPlaceholder for RGB + Near-Infrared data.")
        
        print(f"Created sample multimodal data structure in {data_root}")
    
    print("Download completed!")


if __name__ == '__main__':
    main()