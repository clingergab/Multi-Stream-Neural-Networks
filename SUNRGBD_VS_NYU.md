# SUN RGB-D vs NYU Depth V2 Comparison

## Why Switch to SUN RGB-D?

### Dataset Size
| Dataset | Total Images | Train | Test/Val |
|---------|-------------|-------|----------|
| **NYU Depth V2** | 1,449 | 1,159 (80%) | 290 (20%) |
| **SUN RGB-D** | **10,335** | **5,285** | **5,050** |
| **Improvement** | **7.1x more** | **4.6x more** | **17.4x more** |

### Scene Classification
| Dataset | Scene Classes | Class Balance | Smallest Class |
|---------|--------------|---------------|----------------|
| **NYU Depth V2** | 27 | 192x imbalance | 2 samples (indoor_balcony) |
| **SUN RGB-D** | **37** | Unknown (likely better) | Unknown |

### Key Advantages of SUN RGB-D

1. **7x More Data** (10,335 vs 1,449 images)
   - Enough data for deep learning
   - Better train/test split (5,285 / 5,050)
   - Less overfitting risk

2. **Official Train/Test Split**
   - NYU has no official split
   - SUN RGB-D has standard 5,285 / 5,050 split
   - Reproducible results

3. **Multiple Sensors**
   - NYU: Only Microsoft Kinect
   - SUN RGB-D: 4 different sensors
   - Better generalization

4. **Includes NYU Data**
   - SUN RGB-D contains NYU depth v2
   - Also includes Berkeley B3DO and SUN3D
   - Comprehensive indoor RGB-D dataset

5. **Better for Scene Classification**
   - 37 scene categories (vs 27)
   - Designed for scene understanding
   - Proper annotations

## Download Instructions

### Method 1: Download Script (Recommended)
```bash
./download_sunrgbd.sh
```

### Method 2: Manual Download
```bash
mkdir -p data/sunrgbd
cd data/sunrgbd

# Download dataset (~37GB)
wget http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
wget http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip

# Extract
unzip SUNRGBD.zip
unzip SUNRGBDtoolbox.zip
```

### Method 3: Hugging Face (if available)
Check for SUN RGB-D on HuggingFace datasets.

## Dataset Structure

After download, you'll have:

```
data/sunrgbd/
├── SUNRGBD/           # Main dataset
│   ├── kv1/          # Kinect v1 data
│   ├── kv2/          # Kinect v2 data
│   ├── realsense/    # Intel RealSense data
│   └── xtion/        # ASUS Xtion data
│
└── SUNRGBDtoolbox/    # Annotations & splits
    ├── traintestSUNRGBD/
    │   ├── allsplit.mat     # Train/test splits
    │   └── ...
    └── Metadata/
        └── SUNRGBD2Dseg.mat # Scene labels
```

## Next Steps

### 1. Create SUN RGB-D DataLoader
```python
# src/data_utils/sunrgbd_dataset.py
class SUNRGBDDataset(Dataset):
    def __init__(self, data_dir, split='train', ...):
        # Load train/test split from allsplit.mat
        # Load scene labels from SUNRGBD2Dseg.mat
        # Load RGB + Depth images
        ...
```

### 2. Update Model Config
```python
# Change from:
num_classes = 27  # NYU

# To:
num_classes = 37  # SUN RGB-D
```

### 3. Expected Improvements
- **Training stability** - 5,285 train images (vs 1,159)
- **Validation reliability** - 5,050 test images (vs 290)
- **Better accuracy** - More data = better learning
- **Less overfitting** - Larger, more diverse dataset
- **Reproducible results** - Official splits

## Citation

If you use SUN RGB-D, cite:
```
@inproceedings{song2015sun,
  title={Sun rgb-d: A rgb-d scene understanding benchmark suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={CVPR},
  year={2015}
}
```

## Comparison with NYU Issues

| Issue with NYU | Fixed in SUN RGB-D |
|---------------|-------------------|
| Only 1,449 images | ✅ 10,335 images (7x more) |
| 192x class imbalance | ✅ Likely more balanced |
| No official train/test split | ✅ Official 5,285/5,050 split |
| Val set too small (290) | ✅ Large test set (5,050) |
| 6 classes with ≤4 samples | ✅ Better distribution expected |
| Severe overfitting | ✅ More data reduces overfitting |

## Decision: Switch to SUN RGB-D ✅

**Recommendation: SWITCH to SUN RGB-D for all the reasons above!**
