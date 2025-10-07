"""
Preprocess SUN RGB-D dataset into 15 categories.

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
    labels.txt  # One label per line (0-14)
  val/
    rgb/
    depth/
    labels.txt

Also creates metadata files with class names and statistics.
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm
from sunrgbd_15_category_mapping import map_raw_scene_to_15, SUNRGBD_15_CATEGORIES, get_class_idx

# Paths
SUNRGBD_BASE = "data/sunrgbd/SUNRGBD"
OUTPUT_BASE = "data/sunrgbd_15"
TOOLBOX_PATH = "data/sunrgbd/SUNRGBDtoolbox"

# Use official train/test split from toolbox
TRAIN_SPLIT_FILE = os.path.join(TOOLBOX_PATH, "traintestSUNRGBD/allsplit.mat")

def load_official_split():
    """Load official train/test split from SUNRGBDtoolbox."""
    try:
        import scipy.io as sio
        split_data = sio.loadmat(TRAIN_SPLIT_FILE)

        # Extract train and test indices
        # Note: MATLAB indices are 1-based, need to convert to 0-based
        train_indices = split_data['alltrain'].flatten() - 1
        test_indices = split_data['alltest'].flatten() - 1

        return set(train_indices.tolist()), set(test_indices.tolist())
    except Exception as e:
        print(f"Warning: Could not load official split: {e}")
        print("Will use random 80/20 split instead")
        return None, None

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
        List of tuples: (sample_dir, raw_scene, mapped_scene, class_idx)
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

def create_preprocessed_dataset(samples, train_indices=None, test_indices=None):
    """
    Create preprocessed dataset with proper train/val split.

    Args:
        samples: List of sample tuples
        train_indices: Set of training indices (if using official split)
        test_indices: Set of test indices (if using official split)
    """
    # Create output directories
    for split in ['train', 'val']:
        for subdir in ['rgb', 'depth']:
            os.makedirs(os.path.join(OUTPUT_BASE, split, subdir), exist_ok=True)

    # Split samples
    if train_indices is not None and test_indices is not None:
        print("Using official train/test split from SUNRGBDtoolbox")
        train_samples = [samples[i] for i in train_indices if i < len(samples)]
        val_samples = [samples[i] for i in test_indices if i < len(samples)]
    else:
        print("Using random 80/20 split with stratification")
        # Group by class for stratified split
        from collections import defaultdict
        class_samples = defaultdict(list)
        for idx, sample in enumerate(samples):
            class_idx = sample[3]
            class_samples[class_idx].append(idx)

        train_indices_list = []
        val_indices_list = []

        # Shuffle and split each class
        np.random.seed(42)
        for class_idx, indices in class_samples.items():
            indices = np.array(indices)
            np.random.shuffle(indices)
            split_point = int(len(indices) * 0.8)
            train_indices_list.extend(indices[:split_point])
            val_indices_list.extend(indices[split_point:])

        train_samples = [samples[i] for i in train_indices_list]
        val_samples = [samples[i] for i in val_indices_list]

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Process train split
    print("\nProcessing training set...")
    train_labels = []
    for idx, (sample_dir, raw_scene, mapped_scene, class_idx, rgb_path, depth_path) in enumerate(tqdm(train_samples)):
        # Copy RGB image
        rgb_out = os.path.join(OUTPUT_BASE, 'train', 'rgb', f'{idx:05d}.png')
        img = Image.open(rgb_path).convert('RGB')
        img.save(rgb_out)

        # Copy depth image
        depth_out = os.path.join(OUTPUT_BASE, 'train', 'depth', f'{idx:05d}.png')
        depth = Image.open(depth_path)
        depth.save(depth_out)

        # Record label
        train_labels.append(class_idx)

    # Save train labels
    with open(os.path.join(OUTPUT_BASE, 'train', 'labels.txt'), 'w') as f:
        for label in train_labels:
            f.write(f"{label}\n")

    # Process val split
    print("\nProcessing validation set...")
    val_labels = []
    for idx, (sample_dir, raw_scene, mapped_scene, class_idx, rgb_path, depth_path) in enumerate(tqdm(val_samples)):
        # Copy RGB image
        rgb_out = os.path.join(OUTPUT_BASE, 'val', 'rgb', f'{idx:05d}.png')
        img = Image.open(rgb_path).convert('RGB')
        img.save(rgb_out)

        # Copy depth image
        depth_out = os.path.join(OUTPUT_BASE, 'val', 'depth', f'{idx:05d}.png')
        depth = Image.open(depth_path)
        depth.save(depth_out)

        # Record label
        val_labels.append(class_idx)

    # Save val labels
    with open(os.path.join(OUTPUT_BASE, 'val', 'labels.txt'), 'w') as f:
        for label in val_labels:
            f.write(f"{label}\n")

    # Save metadata
    metadata = {
        'num_classes': 15,
        'class_names': SUNRGBD_15_CATEGORIES,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_distribution': dict(Counter(train_labels)),
        'val_distribution': dict(Counter(val_labels)),
    }

    # Write class names
    with open(os.path.join(OUTPUT_BASE, 'class_names.txt'), 'w') as f:
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"{idx}: {name}\n")

    # Write statistics
    with open(os.path.join(OUTPUT_BASE, 'dataset_info.txt'), 'w') as f:
        f.write("SUN RGB-D 15-Category Dataset\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of classes: {metadata['num_classes']}\n")
        f.write(f"Train samples: {metadata['train_samples']}\n")
        f.write(f"Val samples: {metadata['val_samples']}\n\n")

        f.write("Class names:\n")
        for idx, name in enumerate(SUNRGBD_15_CATEGORIES):
            f.write(f"  {idx:2d}: {name}\n")

        f.write("\nTrain distribution:\n")
        train_counter = Counter(train_labels)
        for class_idx in range(15):
            class_name = SUNRGBD_15_CATEGORIES[class_idx]
            count = train_counter.get(class_idx, 0)
            f.write(f"  {class_idx:2d} {class_name:20s}: {count:5d}\n")

        f.write("\nVal distribution:\n")
        val_counter = Counter(val_labels)
        for class_idx in range(15):
            class_name = SUNRGBD_15_CATEGORIES[class_idx]
            count = val_counter.get(class_idx, 0)
            f.write(f"  {class_idx:2d} {class_name:20s}: {count:5d}\n")

    print(f"\n✓ Dataset preprocessed successfully!")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Train: {len(train_samples)} samples")
    print(f"Val: {len(val_samples)} samples")

def main():
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

    # Load official split (if available)
    train_indices, test_indices = load_official_split()

    # Create preprocessed dataset
    create_preprocessed_dataset(samples, train_indices, test_indices)

if __name__ == "__main__":
    main()
