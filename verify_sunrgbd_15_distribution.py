"""
Verify the sample distribution for SUN RGB-D with 15 merged categories.
"""

import os
from collections import Counter
from sunrgbd_15_category_mapping import map_raw_scene_to_15, SUNRGBD_15_CATEGORIES

# Path to SUN RGB-D dataset
sunrgbd_base = "data/sunrgbd/SUNRGBD"

# Collect all scene labels
all_scenes = []
scene_paths = []

print("Scanning SUN RGB-D dataset...")
for root, dirs, files in os.walk(sunrgbd_base):
    if 'scene.txt' in files:
        scene_file = os.path.join(root, 'scene.txt')
        try:
            with open(scene_file, 'r') as f:
                raw_scene = f.read().strip()
                # Map to 15 categories
                mapped_scene = map_raw_scene_to_15(raw_scene)
                if mapped_scene is not None:
                    all_scenes.append(mapped_scene)
                    scene_paths.append(root)
        except Exception as e:
            print(f"Error reading {scene_file}: {e}")

print(f"\nTotal samples found: {len(all_scenes)}")

# Count distribution
scene_counts = Counter(all_scenes)

print("\n" + "="*80)
print("SUN RGB-D 15-Category Distribution:")
print("="*80)
print(f"{'Category':<25} {'Count':>10} {'Percentage':>12} {'Bar':>30}")
print("-"*80)

total_samples = len(all_scenes)
sorted_counts = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)

for scene, count in sorted_counts:
    percentage = (count / total_samples) * 100
    bar_length = int(percentage / 2)  # Scale to fit 50 chars max
    bar = '█' * bar_length
    print(f"{scene:<25} {count:>10} {percentage:>11.2f}% {bar}")

print("-"*80)
print(f"{'TOTAL':<25} {total_samples:>10} {100.0:>11.2f}%")

# Calculate class imbalance
max_samples = max(scene_counts.values())
min_samples = min(scene_counts.values())
imbalance_ratio = max_samples / min_samples

print(f"\nClass Imbalance Ratio: {imbalance_ratio:.1f}x ({max_samples} / {min_samples})")

# Check if all 15 categories are present
missing_categories = set(SUNRGBD_15_CATEGORIES) - set(scene_counts.keys())
if missing_categories:
    print(f"\nWARNING: Missing categories: {missing_categories}")
else:
    print(f"\n✓ All 15 categories present in dataset")

# Show min samples category
min_category = min(scene_counts.items(), key=lambda x: x[1])
print(f"Smallest class: {min_category[0]} with {min_category[1]} samples")

# Show max samples category
max_category = max(scene_counts.items(), key=lambda x: x[1])
print(f"Largest class: {max_category[0]} with {max_category[1]} samples")

print("\n" + "="*80)
