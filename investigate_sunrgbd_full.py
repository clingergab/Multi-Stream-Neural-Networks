"""
Complete investigation of SUN RGB-D dataset structure.
"""

import os
import json
from collections import Counter, defaultdict
import scipy.io as sio

print("=" * 80)
print("COMPLETE SUN RGB-D INVESTIGATION")
print("=" * 80)

# 1. Check top-level structure
print("\n1. TOP-LEVEL STRUCTURE")
print("-" * 80)
base_path = "data/sunrgbd/SUNRGBD"
sensors = os.listdir(base_path)
sensors = [s for s in sensors if not s.startswith('.')]
print(f"Sensor directories: {sensors}")

for sensor in sensors:
    sensor_path = os.path.join(base_path, sensor)
    if os.path.isdir(sensor_path):
        subdirs = [d for d in os.listdir(sensor_path) if os.path.isdir(os.path.join(sensor_path, d)) and not d.startswith('.')]
        print(f"\n{sensor}/")
        print(f"  Subdirectories: {len(subdirs)}")
        if len(subdirs) <= 5:
            print(f"  Names: {subdirs}")
        else:
            print(f"  Sample names: {subdirs[:3]}")

# 2. Count total samples per sensor
print("\n\n2. SAMPLE COUNTS PER SENSOR")
print("-" * 80)
sensor_counts = {}
total_samples = 0

for sensor in sensors:
    sensor_path = os.path.join(base_path, sensor)
    if not os.path.isdir(sensor_path):
        continue

    count = 0
    for root, dirs, files in os.walk(sensor_path):
        if 'scene.txt' in files:
            count += 1

    sensor_counts[sensor] = count
    total_samples += count
    print(f"{sensor:20s}: {count:5d} samples")

print(f"{'TOTAL':20s}: {total_samples:5d} samples")

# 3. Check file structure of one sample
print("\n\n3. SAMPLE FILE STRUCTURE")
print("-" * 80)
sample_path = "data/sunrgbd/SUNRGBD/kv1/NYUdata/NYU0001"
print(f"Examining: {sample_path}")
print(f"\nFiles and directories:")
for item in sorted(os.listdir(sample_path)):
    item_path = os.path.join(sample_path, item)
    if os.path.isdir(item_path):
        subfiles = os.listdir(item_path)
        print(f"  {item}/ ({len(subfiles)} files)")
        if len(subfiles) <= 3:
            for sf in subfiles:
                print(f"    - {sf}")
    else:
        print(f"  {item}")

# 4. Check what's in annotation files
print("\n\n4. ANNOTATION STRUCTURE")
print("-" * 80)
annotation_path = os.path.join(sample_path, "annotation2D3D")
if os.path.exists(annotation_path):
    json_file = os.path.join(annotation_path, "index.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            annotation = json.load(f)
        print(f"Annotation keys: {list(annotation.keys())}")

        # Check for scene info
        if 'sceneCategory' in annotation:
            print(f"Scene category found: {annotation['sceneCategory']}")
        if 'sceneName' in annotation:
            print(f"Scene name found: {annotation['sceneName']}")

# 5. Check intrinsics format
print("\n\n5. CAMERA INTRINSICS")
print("-" * 80)
intrinsics_path = os.path.join(sample_path, "intrinsics.txt")
if os.path.exists(intrinsics_path):
    with open(intrinsics_path, 'r') as f:
        intrinsics = f.read()
    print(intrinsics)

# 6. Collect ALL scene labels from scene.txt files
print("\n\n6. COMPLETE SCENE DISTRIBUTION")
print("-" * 80)
scenes = []
scene_paths = {}

for sensor in sensors:
    sensor_path = os.path.join(base_path, sensor)
    if not os.path.isdir(sensor_path):
        continue

    for root, dirs, files in os.walk(sensor_path):
        if 'scene.txt' in files:
            scene_file = os.path.join(root, 'scene.txt')
            with open(scene_file, 'r') as f:
                scene = f.read().strip()
                scenes.append(scene)
                if scene not in scene_paths:
                    scene_paths[scene] = []
                scene_paths[scene].append(root)

counter = Counter(scenes)
print(f"Total samples found: {len(scenes)}")
print(f"Unique scene types: {len(counter)}")

print(f"\nScene distribution (all {len(counter)} scenes):")
for scene, count in counter.most_common():
    pct = (count / len(scenes)) * 100
    print(f"  {scene:30s}: {count:5d} ({pct:5.1f}%)")

# 7. Check metadata for scene categories
print("\n\n7. CHECKING METADATA FOR OFFICIAL SCENE CATEGORIES")
print("-" * 80)

# Check if there's a mapping to 19 or other number of classes
meta_path = "data/sunrgbd/SUNRGBDtoolbox/Metadata"
for meta_file in os.listdir(meta_path):
    if meta_file.endswith('.mat'):
        try:
            print(f"\nChecking {meta_file}...")
            mat = sio.loadmat(os.path.join(meta_path, meta_file))
            keys = [k for k in mat.keys() if not k.startswith('__')]
            print(f"  Keys: {keys}")

            for key in keys:
                data = mat[key]
                if hasattr(data, 'shape'):
                    print(f"  {key} shape: {data.shape}")
                if hasattr(data, 'dtype'):
                    if data.dtype.names:
                        print(f"  {key} fields: {data.dtype.names}")
        except Exception as e:
            print(f"  Could not load: {e}")

# 8. Check train/test split
print("\n\n8. TRAIN/TEST SPLIT ANALYSIS")
print("-" * 80)
split_data = sio.loadmat('data/sunrgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')
train_paths = split_data['alltrain'].flatten()
test_paths = split_data['alltest'].flatten()

print(f"Train samples: {len(train_paths)}")
print(f"Test samples: {len(test_paths)}")

# Extract scene from path and count distribution
def extract_scene_from_path(path_str):
    # Paths are like: /n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/...
    # We need to map to local path and read scene.txt
    local_path = path_str.replace('/n/fs/sun3d/data/SUNRGBD/', 'data/sunrgbd/SUNRGBD/')
    scene_file = os.path.join(local_path, 'scene.txt')
    if os.path.exists(scene_file):
        with open(scene_file, 'r') as f:
            return f.read().strip()
    return None

train_scenes = []
test_scenes = []

print("\nExtracting scenes from train/test paths (sampling first 100 each)...")
for path in train_paths[:100]:
    scene = extract_scene_from_path(path[0])
    if scene:
        train_scenes.append(scene)

for path in test_paths[:100]:
    scene = extract_scene_from_path(path[0])
    if scene:
        test_scenes.append(scene)

print(f"\nTrain scene distribution (first 100):")
train_counter = Counter(train_scenes)
for scene, count in train_counter.most_common():
    print(f"  {scene:30s}: {count:3d}")

print(f"\nTest scene distribution (first 100):")
test_counter = Counter(test_scenes)
for scene, count in test_counter.most_common():
    print(f"  {scene:30s}: {count:3d}")

# 9. Summary
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nWhat we have:")
print(f"  • Total RGB-D image pairs: {total_samples}")
print(f"  • Sensors: {len(sensors)} types ({', '.join(sensors)})")
print(f"  • Scene categories: {len(counter)} unique types")
print(f"  • Train/test split: {len(train_paths)} / {len(test_paths)}")
print(f"  • Class imbalance: {max(counter.values())}/{min(counter.values())} = {max(counter.values())/min(counter.values()):.0f}x")
print(f"\nTop 5 scenes:")
for scene, count in counter.most_common(5):
    pct = (count / len(scenes)) * 100
    print(f"  {scene:30s}: {count:5d} ({pct:5.1f}%)")

print(f"\n⚠️  Note: Found {len(counter)} scene types, not 19!")
print(f"   The 19 classes might be a different task (object detection)")
print(f"   or a grouped/simplified version of these {len(counter)} scenes.")
