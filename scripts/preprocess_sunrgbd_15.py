"""
Preprocess SUN RGB-D dataset into 15 categories with official train/test split.

Uses the official SUN RGB-D train/test split from SUNRGBDtoolbox/allsplit.mat,
then sub-splits the official training set into train (80%) and val (20%) for
model selection / early stopping. The official test set is preserved as-is.

With --no-val-split, skips the train/val sub-split and outputs all official
training samples as a single train/ split (for k-fold CV in Ray Tune HPO).

Creates a new directory structure:
data/sunrgbd_15/
  train/
    rgb/
      00000.png
      00001.png
      ...
    depth/
      00000.png
      00001.png
      ...
    rgb_tensors.pt    # [N, 3, 416, 544] uint8 pre-resized
    depth_tensors.pt  # [N, 1, 416, 544] uint8 pre-resized
    labels.txt  # One label per line (0-14)
  val/
    rgb/
    depth/
    rgb_tensors.pt
    depth_tensors.pt
    labels.txt
  test/
    rgb/
    depth/
    rgb_tensors.pt
    depth_tensors.pt
    labels.txt

Also creates metadata files with class names and statistics.
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scripts.sunrgbd_15_category_mapping import map_raw_scene_to_15, SUNRGBD_15_CATEGORIES, get_class_idx

# Paths
SUNRGBD_BASE = "data/sunrgbd/SUNRGBD"
OUTPUT_BASE = "data/sunrgbd_15"
TOOLBOX_PATH = "data/sunrgbd/SUNRGBDtoolbox"

# Use official train/test split from toolbox
TRAIN_SPLIT_FILE = os.path.join(TOOLBOX_PATH, "traintestSUNRGBD/allsplit.mat")

# Prefix used in allsplit.mat paths (maps to local SUNRGBD_BASE)
MAT_PATH_PREFIX = "/n/fs/sun3d/data/SUNRGBD/"

# Target resolution for pre-resized tensor files
TARGET_SIZE = (416, 544)


def _create_tensor_files(split_name, num_samples, output_base=None):
    """
    Create pre-resized uint8 tensor files from the PNG images already saved by _process_split.

    Reads PNGs from {output_base}/{split}/rgb/ and depth/, resizes to TARGET_SIZE,
    and saves as uint8 tensors:
      - rgb_tensors.pt:   [N, 3, H, W] uint8
      - depth_tensors.pt: [N, 1, H, W] uint8

    Depth images (16-bit) are scaled to 0-255 uint8 range via: uint8 = (raw / 65535 * 255).

    At load time, the dataset converts back to float32 via: float = tensor / 255.0
    This introduces negligible quantization error (~0.2% for 8-bit) while reducing
    file sizes by ~2x vs float32 and eliminating PNG decode + resize from __getitem__.
    """
    if output_base is None:
        output_base = OUTPUT_BASE
    split_dir = os.path.join(output_base, split_name)
    rgb_dir = os.path.join(split_dir, 'rgb')
    depth_dir = os.path.join(split_dir, 'depth')

    print(f"\nCreating tensor files for {split_name} ({num_samples} samples, {TARGET_SIZE})...")

    # Pre-allocate tensors
    H, W = TARGET_SIZE
    rgb_tensors = torch.empty(num_samples, 3, H, W, dtype=torch.uint8)
    depth_tensors = torch.empty(num_samples, 1, H, W, dtype=torch.uint8)

    for idx in tqdm(range(num_samples)):
        # RGB: load PNG, resize, convert to [3, H, W] uint8
        rgb_path = os.path.join(rgb_dir, f'{idx:05d}.png')
        rgb = Image.open(rgb_path).convert('RGB')
        rgb = transforms.functional.resize(rgb, TARGET_SIZE)
        # to_tensor converts PIL [0,255] uint8 -> [0,1] float; we want uint8 directly
        rgb_arr = np.array(rgb, dtype=np.uint8)  # [H, W, 3]
        rgb_tensors[idx] = torch.from_numpy(rgb_arr).permute(2, 0, 1)  # [3, H, W]

        # Depth: load 16-bit PNG, scale to 0-255 uint8, resize
        depth_path = os.path.join(depth_dir, f'{idx:05d}.png')
        depth = Image.open(depth_path)

        if depth.mode in ('I', 'I;16', 'I;16B'):
            # 16-bit depth: scale to [0, 255] uint8
            depth_arr = np.array(depth, dtype=np.float32)
            depth_arr = np.clip(depth_arr / 65535.0 * 255.0, 0, 255).astype(np.uint8)
            depth = Image.fromarray(depth_arr, mode='L')
        else:
            depth = depth.convert('L')

        depth = transforms.functional.resize(depth, TARGET_SIZE)
        depth_arr = np.array(depth, dtype=np.uint8)  # [H, W]
        depth_tensors[idx] = torch.from_numpy(depth_arr).unsqueeze(0)  # [1, H, W]

    # Orth: generate if orth PNGs exist (from preprocess_orthogonal_clean.py)
    orth_dir = os.path.join(split_dir, 'orth')
    has_orth = os.path.exists(orth_dir) and len(os.listdir(orth_dir)) >= num_samples

    if has_orth:
        orth_tensors = torch.empty(num_samples, 1, H, W, dtype=torch.uint8)
        print(f"  Orth directory found, creating orth_tensors.pt...")
        for idx in tqdm(range(num_samples), desc="  orth"):
            orth_path = os.path.join(orth_dir, f'{idx:05d}.png')
            orth = Image.open(orth_path)
            if orth.mode in ('I', 'I;16', 'I;16B'):
                orth_arr = np.array(orth, dtype=np.float32)
                orth_arr = np.clip(orth_arr / 65535.0 * 255.0, 0, 255).astype(np.uint8)
                orth = Image.fromarray(orth_arr, mode='L')
            else:
                orth = orth.convert('L')
            orth = transforms.functional.resize(orth, TARGET_SIZE)
            orth_arr = np.array(orth, dtype=np.uint8)
            orth_tensors[idx] = torch.from_numpy(orth_arr).unsqueeze(0)

    # Save tensor files
    rgb_path = os.path.join(split_dir, 'rgb_tensors.pt')
    depth_path = os.path.join(split_dir, 'depth_tensors.pt')
    torch.save(rgb_tensors, rgb_path)
    torch.save(depth_tensors, depth_path)

    rgb_mb = os.path.getsize(rgb_path) / (1024 * 1024)
    depth_mb = os.path.getsize(depth_path) / (1024 * 1024)
    print(f"  rgb_tensors.pt:   {rgb_tensors.shape} → {rgb_mb:.1f} MB")
    print(f"  depth_tensors.pt: {depth_tensors.shape} → {depth_mb:.1f} MB")

    if has_orth:
        orth_out_path = os.path.join(split_dir, 'orth_tensors.pt')
        torch.save(orth_tensors, orth_out_path)
        orth_mb = os.path.getsize(orth_out_path) / (1024 * 1024)
        print(f"  orth_tensors.pt:  {orth_tensors.shape} → {orth_mb:.1f} MB")


def load_official_split():
    """
    Load official train/test split from SUNRGBDtoolbox.

    The allsplit.mat file contains string file paths (not numeric indices).
    Each entry is a path like '/n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000065_...-resize'.
    We strip the prefix and trailing slashes to get relative paths that can be
    matched against local sample directories.

    Returns:
        Tuple of (train_paths, test_paths) where each is a set of relative paths.
        Raises FileNotFoundError if the split file doesn't exist.
    """
    if not os.path.exists(TRAIN_SPLIT_FILE):
        raise FileNotFoundError(
            f"Official split file not found: {TRAIN_SPLIT_FILE}\n"
            f"Please ensure the SUNRGBDtoolbox is extracted at {TOOLBOX_PATH}"
        )

    split_data = sio.loadmat(TRAIN_SPLIT_FILE)

    def extract_paths(mat_array):
        """Extract relative paths from nested numpy string arrays."""
        paths = set()
        for entry in mat_array.flatten():
            # Each entry is a numpy array containing a string
            path_str = str(entry.flatten()[0])
            # Strip the server prefix and trailing slashes
            relative = path_str.replace(MAT_PATH_PREFIX, '').rstrip('/')
            paths.add(relative)
        return paths

    train_paths = extract_paths(split_data['alltrain'])
    test_paths = extract_paths(split_data['alltest'])

    print(f"Official split loaded: {len(train_paths)} train, {len(test_paths)} test")

    return train_paths, test_paths


def find_rgb_depth_files(sample_dir):
    """
    Find RGB and depth image files in a sample directory.

    SUN RGB-D has different naming conventions:
    - RGB: image/*.jpg or image/*.png
    - Depth: depth/*.png or depth_bfx/*.png
    """
    rgb_path = None
    depth_path = None

    # Find RGB image
    image_dir = os.path.join(sample_dir, 'image')
    if os.path.exists(image_dir):
        for ext in ['*.jpg', '*.png']:
            rgb_files = list(Path(image_dir).glob(ext))
            if rgb_files:
                rgb_path = str(rgb_files[0])
                break

    # Find depth image
    for depth_subdir in ['depth_bfx', 'depth']:
        depth_dir = os.path.join(sample_dir, depth_subdir)
        if os.path.exists(depth_dir):
            depth_files = list(Path(depth_dir).glob('*.png'))
            if depth_files:
                depth_path = str(depth_files[0])
                break

    return rgb_path, depth_path


def collect_all_samples():
    """
    Collect all valid samples from SUN RGB-D dataset.

    Returns:
        List of tuples: (sample_dir, raw_scene, mapped_scene, class_idx, rgb_path, depth_path)
    """
    samples = []

    print("Scanning SUN RGB-D dataset...")
    for root, dirs, files in os.walk(SUNRGBD_BASE):
        if 'scene.txt' in files:
            scene_file = os.path.join(root, 'scene.txt')
            try:
                # Read raw scene label
                with open(scene_file, 'r') as f:
                    raw_scene = f.read().strip()

                # Map to 15 categories
                mapped_scene = map_raw_scene_to_15(raw_scene)
                if mapped_scene is None:
                    continue  # Skip "idk" and unknown scenes

                # Get class index
                class_idx = get_class_idx(mapped_scene)
                if class_idx is None:
                    continue

                # Verify RGB and depth files exist
                rgb_path, depth_path = find_rgb_depth_files(root)
                if rgb_path is None or depth_path is None:
                    print(f"Warning: Missing files in {root}")
                    continue

                samples.append((root, raw_scene, mapped_scene, class_idx, rgb_path, depth_path))

            except Exception as e:
                print(f"Error processing {scene_file}: {e}")

    return samples


def _process_split(split_name, split_samples, output_base=None):
    """
    Process a single split: copy images and write labels.

    Args:
        split_name: One of 'train', 'val', 'test'
        split_samples: List of sample tuples for this split
        output_base: Output directory (defaults to OUTPUT_BASE)

    Returns:
        List of class labels for this split
    """
    if output_base is None:
        output_base = OUTPUT_BASE

    # Create output directories
    for subdir in ['rgb', 'depth']:
        os.makedirs(os.path.join(output_base, split_name, subdir), exist_ok=True)

    print(f"\nProcessing {split_name} set ({len(split_samples)} samples)...")
    labels = []
    for idx, (sample_dir, raw_scene, mapped_scene, class_idx, rgb_path, depth_path) in enumerate(tqdm(split_samples)):
        # Copy RGB image
        rgb_out = os.path.join(output_base, split_name, 'rgb', f'{idx:05d}.png')
        img = Image.open(rgb_path).convert('RGB')
        img.save(rgb_out)

        # Copy depth image
        depth_out = os.path.join(output_base, split_name, 'depth', f'{idx:05d}.png')
        depth = Image.open(depth_path)
        depth.save(depth_out)

        # Record label
        labels.append(class_idx)

    # Save labels
    with open(os.path.join(output_base, split_name, 'labels.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

    return labels


def create_preprocessed_dataset(samples, train_paths, test_paths, val_ratio=0.2, seed=42):
    """
    Create preprocessed dataset with official 3-way split (train/val/test).

    The official training set is sub-split into train and val using stratified
    sampling. The official test set is preserved as-is.

    Args:
        samples: List of sample tuples from collect_all_samples()
        train_paths: Set of relative paths for official training samples
        test_paths: Set of relative paths for official test samples
        val_ratio: Fraction of official training set to use for validation
        seed: Random seed for reproducible train/val sub-split
    """
    # Build lookup: relative path -> sample
    path_to_sample = {}
    for sample in samples:
        sample_dir = sample[0]
        rel_path = os.path.relpath(sample_dir, SUNRGBD_BASE)
        path_to_sample[rel_path] = sample

    # Match samples to official splits
    official_train_samples = []
    official_test_samples = []
    unmatched_train = 0
    unmatched_test = 0

    for path in train_paths:
        if path in path_to_sample:
            official_train_samples.append(path_to_sample[path])
        else:
            unmatched_train += 1

    for path in test_paths:
        if path in path_to_sample:
            official_test_samples.append(path_to_sample[path])
        else:
            unmatched_test += 1

    print(f"\nOfficial split matching:")
    print(f"  Train: {len(official_train_samples)} matched ({unmatched_train} filtered by 15-cat mapping)")
    print(f"  Test:  {len(official_test_samples)} matched ({unmatched_test} filtered by 15-cat mapping)")

    # Sub-split official train into train (80%) and val (20%) — stratified by class
    class_groups = defaultdict(list)
    for idx, sample in enumerate(official_train_samples):
        class_idx = sample[3]
        class_groups[class_idx].append(idx)

    np.random.seed(seed)
    train_sub_indices = []
    val_sub_indices = []

    for class_idx in sorted(class_groups.keys()):
        indices = np.array(class_groups[class_idx])
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1.0 - val_ratio))
        train_sub_indices.extend(indices[:split_point])
        val_sub_indices.extend(indices[split_point:])

    train_samples = [official_train_samples[i] for i in train_sub_indices]
    val_samples = [official_train_samples[i] for i in val_sub_indices]
    test_samples = official_test_samples

    print(f"\n3-way split:")
    print(f"  Train: {len(train_samples)} (80% of official train)")
    print(f"  Val:   {len(val_samples)} (20% of official train)")
    print(f"  Test:  {len(test_samples)} (official test set)")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)}")

    # Process all three splits (save PNGs + create pre-resized tensor files)
    train_labels = _process_split('train', train_samples)
    _create_tensor_files('train', len(train_samples))
    val_labels = _process_split('val', val_samples)
    _create_tensor_files('val', len(val_samples))
    test_labels = _process_split('test', test_samples)
    _create_tensor_files('test', len(test_samples))

    # Write class names
    with open(os.path.join(OUTPUT_BASE, 'class_names.txt'), 'w') as f:
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"{idx}: {name}\n")

    # Write statistics
    with open(os.path.join(OUTPUT_BASE, 'dataset_info.txt'), 'w') as f:
        f.write("SUN RGB-D 15-Category Dataset (Official Split)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Split: Official train/test from SUNRGBDtoolbox allsplit.mat\n")
        f.write(f"Train/Val sub-split: {1.0 - val_ratio:.0%}/{val_ratio:.0%} stratified (seed={seed})\n\n")
        f.write(f"Number of classes: 15\n")
        f.write(f"Train samples: {len(train_samples)}\n")
        f.write(f"Val samples: {len(val_samples)}\n")
        f.write(f"Test samples: {len(test_samples)}\n")
        f.write(f"Total: {len(train_samples) + len(val_samples) + len(test_samples)}\n\n")

        f.write("Class names:\n")
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"  {idx:2d}: {name}\n")

        for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
            f.write(f"\n{split_name} distribution:\n")
            counter = Counter(split_labels)
            for class_idx in range(15):
                class_name = SUNRGBD_15_CATEGORIES[class_idx]
                count = counter.get(class_idx, 0)
                f.write(f"  {class_idx:2d} {class_name:20s}: {count:5d}\n")

    print(f"\n✓ Dataset preprocessed successfully!")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Train: {len(train_samples)} samples")
    print(f"Val: {len(val_samples)} samples")
    print(f"Test: {len(test_samples)} samples")


def create_trainval_dataset(samples, train_paths, test_paths, output_base):
    """
    Create preprocessed dataset with 2-way split (train/test only, no val).

    All official training samples go into train/ (no sub-split). This is used
    for k-fold cross-validation in Ray Tune HPO, where each trial creates its
    own train/val split from the full training set.

    Args:
        samples: List of sample tuples from collect_all_samples()
        train_paths: Set of relative paths for official training samples
        test_paths: Set of relative paths for official test samples
        output_base: Output directory path
    """
    # Build lookup: relative path -> sample
    path_to_sample = {}
    for sample in samples:
        sample_dir = sample[0]
        rel_path = os.path.relpath(sample_dir, SUNRGBD_BASE)
        path_to_sample[rel_path] = sample

    # Match samples to official splits
    official_train_samples = []
    official_test_samples = []
    unmatched_train = 0
    unmatched_test = 0

    for path in train_paths:
        if path in path_to_sample:
            official_train_samples.append(path_to_sample[path])
        else:
            unmatched_train += 1

    for path in test_paths:
        if path in path_to_sample:
            official_test_samples.append(path_to_sample[path])
        else:
            unmatched_test += 1

    print(f"\nOfficial split matching:")
    print(f"  Train: {len(official_train_samples)} matched ({unmatched_train} filtered by 15-cat mapping)")
    print(f"  Test:  {len(official_test_samples)} matched ({unmatched_test} filtered by 15-cat mapping)")

    print(f"\n2-way split (no val):")
    print(f"  Train: {len(official_train_samples)} (ALL official train)")
    print(f"  Test:  {len(official_test_samples)} (official test set)")
    print(f"  Total: {len(official_train_samples) + len(official_test_samples)}")

    # Process both splits (save PNGs + create pre-resized tensor files)
    train_labels = _process_split('train', official_train_samples, output_base=output_base)
    _create_tensor_files('train', len(official_train_samples), output_base=output_base)
    test_labels = _process_split('test', official_test_samples, output_base=output_base)
    _create_tensor_files('test', len(official_test_samples), output_base=output_base)

    # Write class names
    with open(os.path.join(output_base, 'class_names.txt'), 'w') as f:
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"{idx}: {name}\n")

    # Write statistics
    with open(os.path.join(output_base, 'dataset_info.txt'), 'w') as f:
        f.write("SUN RGB-D 15-Category Dataset (Official Split, No Val)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Split: Official train/test from SUNRGBDtoolbox allsplit.mat\n")
        f.write(f"No train/val sub-split (for k-fold CV)\n\n")
        f.write(f"Number of classes: 15\n")
        f.write(f"Train samples: {len(official_train_samples)}\n")
        f.write(f"Test samples: {len(official_test_samples)}\n")
        f.write(f"Total: {len(official_train_samples) + len(official_test_samples)}\n\n")

        f.write("Class names:\n")
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"  {idx:2d}: {name}\n")

        for split_name, split_labels in [('Train', train_labels), ('Test', test_labels)]:
            f.write(f"\n{split_name} distribution:\n")
            counter = Counter(split_labels)
            for class_idx in range(15):
                class_name = SUNRGBD_15_CATEGORIES[class_idx]
                count = counter.get(class_idx, 0)
                f.write(f"  {class_idx:2d} {class_name:20s}: {count:5d}\n")

    print(f"\n\u2713 Dataset preprocessed successfully!")
    print(f"Output directory: {output_base}")
    print(f"Train: {len(official_train_samples)} samples (all official train)")
    print(f"Test: {len(official_test_samples)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SUN RGB-D dataset into 15 categories."
    )
    parser.add_argument(
        "--no-val-split",
        action="store_true",
        help="Skip train/val sub-split. Output all official training samples "
             "as a single train/ split (for k-fold CV).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to data/sunrgbd_15 (normal) or "
             "data/sunrgbd_15_trainval (--no-val-split).",
    )
    args = parser.parse_args()

    # Collect all samples
    samples = collect_all_samples()
    print(f"\nFound {len(samples)} valid samples")

    # Check class distribution
    class_counter = Counter([s[3] for s in samples])
    print("\nClass distribution:")
    for class_idx in range(15):
        class_name = SUNRGBD_15_CATEGORIES[class_idx]
        count = class_counter.get(class_idx, 0)
        print(f"  {class_idx:2d} {class_name:20s}: {count:5d}")

    # Load official split
    train_paths, test_paths = load_official_split()

    if args.no_val_split:
        output_base = args.output_dir or "data/sunrgbd_15_trainval"
        create_trainval_dataset(samples, train_paths, test_paths, output_base=output_base)
    else:
        if args.output_dir:
            global OUTPUT_BASE
            OUTPUT_BASE = args.output_dir
        create_preprocessed_dataset(samples, train_paths, test_paths)


if __name__ == "__main__":
    main()
