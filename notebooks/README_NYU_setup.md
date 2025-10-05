# Using Real NYU Depth V2 Data in Test Notebook

## Quick Start

### Option 1: Automatic Download (Recommended)

```bash
# From project root
python3 scripts/download_nyu_depth_v2.py --save-dir ./data

# This will download to: ./data/nyu_depth_v2_labeled.mat (~2.8 GB)
```

### Option 2: Manual Download

```bash
# Download directly
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P ./data/

# Or use curl
curl -o ./data/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

## Enable Real Data in Notebook

Once you've downloaded the dataset:

1. **Open the notebook:** `notebooks/test_nyu_training_pipeline.ipynb`

2. **Find "Option B" section** (around cell 7-8)

3. **Comment out Option A (Synthetic)**:
   ```python
   # COMMENT OUT THIS SECTION
   # train_dataset = SyntheticRGBDDataset(...)
   # val_dataset = SyntheticRGBDDataset(...)
   ```

4. **Uncomment Option B (Real NYU)**:
   ```python
   # UNCOMMENT THIS SECTION
   from src.data_utils.nyu_depth_dataset import create_nyu_dataloaders

   NYU_DATASET_PATH = "./data/nyu_depth_v2_labeled.mat"  # Update if needed

   train_loader, val_loader = create_nyu_dataloaders(
       h5_file_path=NYU_DATASET_PATH,
       batch_size=16,
       num_workers=2,
       target_size=(224, 224),
       num_classes=13
   )
   ```

5. **Run the notebook** - Now using real RGB-D data!

## Dataset Information

### NYU Depth V2 Specs
- **Size:** ~2.8 GB (compressed .mat file)
- **Total Images:** ~1,449 RGB-D pairs
- **Image Resolution:** 640×480 (resized to 224×224 for training)
- **RGB Channels:** 3 (standard RGB)
- **Depth Channels:** 1 (depth map in meters)
- **Task:** Scene classification (13 room types)

### Scene Classes (13 categories)
1. Bathroom
2. Bedroom
3. Bookstore
4. Cafe
5. Classroom
6. Computer Lab
7. Conference Room
8. Corridor
9. Dining Room
10. Home Office
11. Kitchen
12. Living Room
13. Office

### Train/Val Split
- **Training:** 80% (~1,159 images)
- **Validation:** 20% (~290 images)

## Performance Comparison

| Dataset | Samples | Training Time (2 epochs) | Val Accuracy |
|---------|---------|--------------------------|--------------|
| **Synthetic** | 100 train, 20 val | 2-3 min (CPU) | ~10-20% (random) |
| **Real NYU** | ~1,159 train, ~290 val | 10-15 min (CPU), 3-5 min (GPU) | ~30-50% (2 epochs) |

## Memory Requirements

### Synthetic Data
- **RAM:** ~500 MB
- **VRAM (GPU):** ~2 GB

### Real NYU Depth V2
- **RAM:** ~4-6 GB (with num_workers=2)
- **VRAM (GPU):** ~4-6 GB (batch_size=16)
- **Storage:** ~2.8 GB (dataset file)

## Troubleshooting

### Issue: File not found
```python
FileNotFoundError: nyu_depth_v2_labeled.mat
```
**Solution:** Update `NYU_DATASET_PATH` to the correct location

### Issue: Out of Memory
```python
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or num_workers:
```python
train_loader, val_loader = create_nyu_dataloaders(
    h5_file_path=NYU_DATASET_PATH,
    batch_size=8,  # Reduce from 16
    num_workers=0,  # Reduce from 2
    ...
)
```

### Issue: Slow data loading
```python
# Training taking too long
```
**Solution:** Increase num_workers (if you have CPU cores):
```python
num_workers=4  # Use more parallel workers
```

### Issue: h5py not installed
```python
ModuleNotFoundError: No module named 'h5py'
```
**Solution:**
```bash
pip install h5py
```

## Expected Results with Real Data

After 2 epochs with real NYU Depth V2 data:

```
Train Accuracy: 30-40%
Val Accuracy: 25-35%
RGB Pathway: 20-30%
Depth Pathway: 15-25%
```

After 90 epochs (full training):
```
Train Accuracy: 80-90%
Val Accuracy: 70-80%
RGB Pathway: 60-70%
Depth Pathway: 50-60%
```

## Next Steps

Once you've verified the pipeline works with real data:

1. ✅ **Test passed locally** with NYU Depth V2
2. ✅ **Push code to GitHub/Drive**
3. ✅ **Open Google Colab**
4. ✅ **Upload dataset to Colab** or download there
5. ✅ **Train for 90 epochs** on A100 GPU

## Resources

- **Dataset Page:** http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
- **Paper:** "Indoor Segmentation and Support Inference from RGBD Images" (Silberman et al., ECCV 2012)
- **Download Script:** `scripts/download_nyu_depth_v2.py`
- **Dataset Class:** `src/data_utils/nyu_depth_dataset.py`

---

**Ready to test with real data!** 🎉
