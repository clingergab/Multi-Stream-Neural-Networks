# Quick Start - SUN RGB-D Training on Colab

## Step 1: Upload Dataset to Google Drive

Upload the `data/sunrgbd_15/` folder to:
```
Google Drive → My Drive → datasets → sunrgbd_15/
```

**No compression needed!** Just upload the folder directly (4.3 GB).

---

## Step 2: Colab Setup (3 Cells)

### Cell 1: Mount Drive & Copy Dataset
```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

# Copy dataset to local disk (10-20x faster than reading from Drive)
DRIVE_PATH = "/content/drive/MyDrive/datasets/sunrgbd_15"
LOCAL_PATH = "/content/data/sunrgbd_15"

if not Path(LOCAL_PATH).exists():
    print("Copying dataset to local disk (~2-3 min)...")
    !mkdir -p /content/data
    !cp -r {DRIVE_PATH} /content/data/
    print("✅ Done!")

# Verify
train_count = len(list(Path(LOCAL_PATH).glob("train/rgb/*.png")))
print(f"Train: {train_count} samples")
```

### Cell 2: Clone Repo
```python
!git clone https://github.com/YOUR_USERNAME/Multi-Stream-Neural-Networks.git
%cd Multi-Stream-Neural-Networks
```

### Cell 3: Train
```python
!python colab_train_sunrgbd.py
```

---

## That's It! 🚀

**Setup time:** ~5 minutes  
**Expected results:** ~60-65% validation accuracy after 30 epochs

See [SUNRGBD_SETUP_COMPLETE.md](SUNRGBD_SETUP_COMPLETE.md) for full details.
