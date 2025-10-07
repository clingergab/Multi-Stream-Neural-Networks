"""
Check raw NYU dataset structure to verify labels are correct.
"""

import h5py
import numpy as np

print("=" * 80)
print("RAW NYU DATASET INSPECTION")
print("=" * 80)

# Open the HDF5 file directly
h5_path = 'data/content/nyu_depth_v2_labeled.mat'
with h5py.File(h5_path, 'r') as f:
    print("\nAvailable keys in HDF5 file:")
    print(list(f.keys()))

    # Check images
    if 'images' in f:
        images = f['images']
        print(f"\nimages shape: {images.shape}")
        print(f"images dtype: {images.dtype}")

    # Check depths
    if 'depths' in f:
        depths = f['depths']
        print(f"\ndepths shape: {depths.shape}")
        print(f"depths dtype: {depths.dtype}")

    # Check scenes (labels)
    if 'scenes' in f:
        scenes = f['scenes']
        print(f"\nscenes shape: {scenes.shape}")
        print(f"scenes dtype: {scenes.dtype}")

        # Load all scene labels
        if scenes.ndim == 2:
            all_scenes = np.array(scenes[0, :])
        else:
            all_scenes = np.array(scenes[:])

        print(f"\nScene label analysis:")
        print(f"  Total samples: {len(all_scenes)}")
        print(f"  Min label: {all_scenes.min()}")
        print(f"  Max label: {all_scenes.max()}")
        print(f"  Unique labels: {np.unique(all_scenes)}")
        print(f"  Number of unique labels: {len(np.unique(all_scenes))}")

        # Count distribution
        unique, counts = np.unique(all_scenes, return_counts=True)
        print(f"\nRaw label distribution:")
        for label, count in zip(unique, counts):
            pct = (count / len(all_scenes)) * 100
            print(f"  Label {label}: {count} samples ({pct:.1f}%)")

        # Check if this is the actual scene classification or something else
        print(f"\n⚠️  IMPORTANT: Are these scene labels or something else?")
        print(f"  If min={all_scenes.min()}, max={all_scenes.max()}, there are {all_scenes.max() - all_scenes.min() + 1} possible values")

    # Check for names
    if 'names' in f:
        names = f['names']
        print(f"\nnames found: {names}")

    # Check for scene names
    if 'sceneTypes' in f:
        scene_types = f['sceneTypes']
        print(f"\nsceneTypes shape: {scene_types.shape}")
        print(f"sceneTypes dtype: {scene_types.dtype}")

        # Try to read scene type names
        if scene_types.ndim == 2:
            print(f"\nAttempting to decode scene type names:")
            for i in range(min(scene_types.shape[1], 30)):
                try:
                    # HDF5 in MATLAB format stores strings as references
                    refs = scene_types[0, i]
                    if isinstance(refs, h5py.Reference):
                        name = ''.join(chr(c[0]) for c in f[refs][:])
                        print(f"  {i}: {name}")
                except:
                    pass

    # List ALL keys to see what we're missing
    print(f"\n\nFULL HDF5 STRUCTURE:")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Group: {name}")

    f.visititems(print_structure)
