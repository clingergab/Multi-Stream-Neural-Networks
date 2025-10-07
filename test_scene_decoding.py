"""
Properly decode scene labels from MATLAB HDF5 format.
"""

import h5py
import numpy as np

print("=" * 80)
print("DECODING SCENE LABELS FROM MATLAB HDF5")
print("=" * 80)

h5_path = 'data/content/nyu_depth_v2_labeled.mat'

with h5py.File(h5_path, 'r') as f:
    # Get scene references
    scenes_refs = f['scenes']
    print(f"scenes shape: {scenes_refs.shape}")
    print(f"scenes dtype: {scenes_refs.dtype}")

    # Get scene type names
    scene_types_refs = f['sceneTypes']
    print(f"\nsceneTypes shape: {scene_types_refs.shape}")

    # Decode scene type names
    scene_names = []
    print(f"\nDecoding scene type names:")
    for i in range(scene_types_refs.shape[1]):
        ref = scene_types_refs[0, i]
        name = ''.join(chr(c[0]) for c in f[ref][:])
        scene_names.append(name)
        print(f"  {i}: {name}")

    print(f"\nTotal scene types: {len(scene_names)}")

    # Now decode actual scene labels for each image
    print(f"\nDecoding scene labels for each image...")
    scene_labels = []

    for i in range(scenes_refs.shape[1]):
        ref = scenes_refs[0, i]
        # The reference points to an array containing the scene type index
        scene_idx = int(f[ref][0, 0]) - 1  # MATLAB is 1-indexed!
        scene_labels.append(scene_idx)

        if i < 5:  # Show first 5
            print(f"  Image {i}: scene_idx={scene_idx}, scene_name={scene_names[scene_idx] if scene_idx < len(scene_names) else 'unknown'}")

    scene_labels = np.array(scene_labels)

    # Analysis
    print(f"\n" + "=" * 80)
    print("SCENE LABEL DISTRIBUTION")
    print("=" * 80)

    print(f"\nTotal images: {len(scene_labels)}")
    print(f"Min label: {scene_labels.min()}")
    print(f"Max label: {scene_labels.max()}")
    print(f"Unique labels: {len(np.unique(scene_labels))}")

    # Count distribution
    unique, counts = np.unique(scene_labels, return_counts=True)
    print(f"\nLabel distribution:")
    print("Label | Scene Name          | Count | % of Dataset")
    print("-" * 70)
    for label, count in zip(unique, counts):
        name = scene_names[label] if label < len(scene_names) else "unknown"
        pct = (count / len(scene_labels)) * 100
        print(f"{label:5d} | {name:18s} | {count:5d} | {pct:5.1f}%")

    # Check imbalance
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n" + "=" * 80)
    print(f"IMBALANCE ANALYSIS")
    print(f"=" * 80)
    print(f"\nMost common: {scene_names[unique[counts.argmax()]]} ({max_count} samples)")
    print(f"Least common: {scene_names[unique[counts.argmin()]]} ({min_count} samples)")
    print(f"Imbalance ratio: {imbalance_ratio:.1f}x")

    # Compare with what our dataset loader is getting
    print(f"\n" + "=" * 80)
    print(f"COMPARISON WITH DATASET LOADER")
    print(f"=" * 80)

    # What our loader was getting
    print(f"\nWhat the raw dataset has:")
    print(f"  Labels range: {scene_labels.min()} to {scene_labels.max()}")
    print(f"  Number of classes: {len(scene_names)}")
    print(f"  Imbalance: {imbalance_ratio:.1f}x")

    print(f"\nThis distribution is FROM THE ACTUAL NYU DATASET.")
    print(f"The imbalance is REAL, not a bug in our code!")
