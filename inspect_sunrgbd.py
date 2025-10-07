"""
Inspect SUN RGB-D dataset structure and metadata.
"""

import scipy.io as sio
import os
from collections import Counter

print("=" * 80)
print("SUN RGB-D DATASET INSPECTION")
print("=" * 80)

# Load train/test split
split_file = 'data/sunrgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'
print(f"\nLoading split file: {split_file}")
split_data = sio.loadmat(split_file)

print(f"\nKeys in split file: {split_data.keys()}")

# Extract train and test indices
if 'trainvalsplit' in split_data:
    split_info = split_data['trainvalsplit']
    print(f"\ntrainvalsplit shape: {split_info.shape}")
    print(f"trainvalsplit dtype: {split_info.dtype}")

if 'alltrain' in split_data:
    train_indices = split_data['alltrain'].flatten()
    print(f"\nTrain set size: {len(train_indices)}")
    print(f"Train indices range: {train_indices.min()} to {train_indices.max()}")

if 'alltest' in split_data:
    test_indices = split_data['alltest'].flatten()
    print(f"Test set size: {len(test_indices)}")
    print(f"Test indices range: {test_indices.min()} to {test_indices.max()}")

# Load metadata for scene labels
metadata_file = 'data/sunrgbd/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
print(f"\n\nLoading metadata: {metadata_file}")

if os.path.exists(metadata_file):
    metadata = sio.loadmat(metadata_file)
    print(f"\nKeys in metadata: {list(metadata.keys())}")

    if 'SUNRGBD2Dseg' in metadata:
        data = metadata['SUNRGBD2Dseg'][0]
        print(f"\nTotal samples in metadata: {len(data)}")

        # Extract scene categories
        scene_names = []
        for i, item in enumerate(data[:10]):  # Check first 10
            print(f"\nSample {i} fields: {item.dtype.names}")
            if 'scene' in item.dtype.names:
                scene = item['scene']
                if scene.size > 0:
                    scene_name = scene[0][0] if scene.ndim > 0 else str(scene)
                    scene_names.append(scene_name)
                    print(f"  Scene: {scene_name}")
else:
    print(f"Metadata file not found!")

# Try alternative metadata location
alt_metadata = 'data/sunrgbd/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'
if os.path.exists(alt_metadata):
    print(f"\n\nFound alternative metadata: {alt_metadata}")
    alt_data = sio.loadmat(alt_metadata)
    print(f"Keys: {list(alt_data.keys())}")

    if 'SUNRGBDMeta' in alt_data:
        meta = alt_data['SUNRGBDMeta'][0]
        print(f"\nTotal samples: {len(meta)}")

        # Check structure
        print(f"Sample 0 fields: {meta[0].dtype.names}")

        # Extract scene categories
        scenes = []
        for i in range(len(meta)):
            if 'scene' in meta[i].dtype.names:
                scene_data = meta[i]['scene']
                if scene_data.size > 0:
                    scene_name = scene_data[0][0] if isinstance(scene_data[0], np.ndarray) else str(scene_data[0])
                    scenes.append(scene_name)

        print(f"\nExtracted {len(scenes)} scene labels")
        scene_counts = Counter(scenes)
        print(f"Unique scenes: {len(scene_counts)}")
        print(f"\nScene distribution:")
        for scene, count in scene_counts.most_common(20):
            pct = (count / len(scenes)) * 100
            print(f"  {scene:30s}: {count:4d} ({pct:5.1f}%)")

print("\n" + "=" * 80)
print("Checking actual file structure...")
print("=" * 80)

# Count actual images
sensors = ['kv1', 'kv2', 'realsense', 'xtion']
total_images = 0

for sensor in sensors:
    sensor_path = f'data/sunrgbd/SUNRGBD/{sensor}'
    if os.path.exists(sensor_path):
        # Count subdirectories (each is a scene)
        scenes = [d for d in os.listdir(sensor_path) if os.path.isdir(os.path.join(sensor_path, d)) and not d.startswith('.')]
        print(f"\n{sensor}: {len(scenes)} scene directories")
        total_images += len(scenes)

print(f"\nTotal scene directories: {total_images}")
