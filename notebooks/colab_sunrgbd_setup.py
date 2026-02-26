"""
SUN RGB-D Dataset Setup for Colab
Copy this cell to your Colab notebook
"""

from pathlib import Path
import shutil

# Paths
DRIVE_DATASET_PATH = "/content/drive/MyDrive/datasets/sunrgbd_15"
LOCAL_DATASET_PATH = "/content/data/sunrgbd_15"  # Local disk (FAST)

print("=" * 80)
print("SUN RGB-D 15-CATEGORY DATASET SETUP")
print("=" * 80)

# Check if already on local disk
if Path(LOCAL_DATASET_PATH).exists():
    print(f"✅ Dataset already on local disk: {LOCAL_DATASET_PATH}")

    # Verify structure (3-way split: train/val/test using official SUN RGB-D split)
    train_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("train/rgb/*.png")))
    val_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("val/rgb/*.png")))
    test_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("test/rgb/*.png")))

    print(f"   Train samples: {train_rgb}")
    print(f"   Val samples: {val_rgb}")
    print(f"   Test samples: {test_rgb}")

    if train_rgb > 0 and val_rgb > 0 and test_rgb > 0:
        print(f"   ✅ Dataset complete! ({train_rgb + val_rgb + test_rgb} total)")
    else:
        print(f"   ⚠ Dataset incomplete, will re-copy from Drive")
        shutil.rmtree(LOCAL_DATASET_PATH)

# Copy from Drive to local disk
if not Path(LOCAL_DATASET_PATH).exists():
    if Path(DRIVE_DATASET_PATH).exists():
        print(f"\n📁 Found dataset on Drive: {DRIVE_DATASET_PATH}")
        print(f"📥 Copying to local disk for 10-20x faster training...")
        print(f"   This takes ~2-3 minutes but saves 60+ minutes during training!")

        # Create parent directory
        Path(LOCAL_DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)

        # Copy with progress (using cp is faster than shutil for large dirs)
        import subprocess
        result = subprocess.run(
            ['cp', '-r', DRIVE_DATASET_PATH, LOCAL_DATASET_PATH],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✅ Dataset copied to local disk")

            # Verify
            train_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("train/rgb/*.png")))
            val_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("val/rgb/*.png")))
            test_rgb = len(list(Path(LOCAL_DATASET_PATH).glob("test/rgb/*.png")))

            print(f"   Train samples: {train_rgb}")
            print(f"   Val samples: {val_rgb}")
            print(f"   Test samples: {test_rgb}")

            if train_rgb > 0 and val_rgb > 0 and test_rgb > 0:
                print(f"   ✅ All samples verified! ({train_rgb + val_rgb + test_rgb} total)")
            else:
                print(f"   ⚠ Warning: Missing samples in one or more splits")
        else:
            print(f"✗ Copy failed: {result.stderr}")
            raise RuntimeError("Failed to copy dataset from Drive")
    else:
        print(f"\n✗ Dataset not found on Drive: {DRIVE_DATASET_PATH}")
        print(f"\n📤 Please upload your preprocessed dataset to Google Drive:")
        print(f"\n   1. On your local machine, locate: data/sunrgbd_15/")
        print(f"   2. Upload the entire 'sunrgbd_15' folder to:")
        print(f"      Google Drive → My Drive → datasets/")
        print(f"\n   Expected structure in Drive:")
        print(f"      My Drive/")
        print(f"      └── datasets/")
        print(f"          └── sunrgbd_15/")
        print(f"              ├── train/")
        print(f"              │   ├── rgb/       (training images)")
        print(f"              │   ├── depth/     (training images)")
        print(f"              │   └── labels.txt")
        print(f"              ├── val/")
        print(f"              │   ├── rgb/       (validation images)")
        print(f"              │   ├── depth/     (validation images)")
        print(f"              │   └── labels.txt")
        print(f"              ├── test/")
        print(f"              │   ├── rgb/       (official test images)")
        print(f"              │   ├── depth/     (official test images)")
        print(f"              │   └── labels.txt")
        print(f"              ├── class_names.txt")
        print(f"              └── dataset_info.txt")
        print(f"\n   Then re-run this cell.")
        raise FileNotFoundError(f"Dataset not found: {DRIVE_DATASET_PATH}")

print("\n" + "=" * 80)
print(f"✅ Dataset ready at: {LOCAL_DATASET_PATH}")
print("=" * 80)
print("\nYou can now proceed with training!")
print(f"\nUsage in training code:")
print(f"  train_loader, val_loader, test_loader = get_sunrgbd_dataloaders(")
print(f"      data_root='{LOCAL_DATASET_PATH}',")
print(f"      batch_size=64,")
print(f"      num_workers=2")
print(f"  )")
