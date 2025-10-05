# MCResNet Training Pipeline Test Notebook

## Purpose

This notebook tests the complete MCResNet training pipeline locally before deploying to Google Colab for NYU Depth V2 training.

## What It Tests

✅ **Data Pipeline:**
- Synthetic RGB-D dataset creation (mimics NYU Depth V2 format)
- DataLoader creation with proper batching
- Synchronized RGB + Depth augmentation

✅ **Model:**
- MCResNet18/50 instantiation
- Model compilation with optimizations
- Forward pass verification

✅ **Training:**
- Full training loop (1-2 epochs)
- Gradient clipping (NEW!)
- Optional cache clearing (NEW!)
- AMP on CUDA
- Learning rate scheduling
- Validation during training

✅ **Evaluation:**
- Model evaluation
- Pathway analysis (RGB vs Depth contributions)
- Predictions & probabilities
- Visualization

## Quick Start

1. **Open the notebook:**
   ```bash
   cd notebooks
   jupyter notebook test_nyu_training_pipeline.ipynb
   ```
   Or open in VS Code with Jupyter extension

2. **Run all cells** (Cmd/Ctrl + Shift + Enter)

3. **Verify all checks pass** at the end

## Expected Runtime

- **CPU:** 2-3 minutes for 2 epochs
- **GPU (CUDA):** < 1 minute for 2 epochs
- **Apple Silicon (MPS):** ~1 minute for 2 epochs

## What to Look For

### ✅ Success Indicators:
- All cells run without errors
- Training loss decreases
- Validation accuracy > random (>7.7% for 13 classes)
- Pathway analysis shows both RGB and Depth contribute
- All verification checks pass at the end

### ⚠️ Warning Signs:
- CUDA out of memory → Reduce BATCH_SIZE
- Loss becomes NaN → Check data normalization
- Val accuracy = random → Expected for synthetic data, OK for testing

## Customization

### Test Different Architectures:
```python
# In cell 5, change:
MODEL_CONFIG = {
    'architecture': 'resnet50',  # or 'resnet18'
    ...
}
```

### Test Larger Datasets:
```python
# In cell 2, increase samples:
train_dataset = SyntheticRGBDDataset(
    num_samples=500,  # Increase from 100
    ...
)
```

### Test Different Optimizations:
```python
# In cell 7, modify:
TRAIN_CONFIG = {
    'epochs': 5,  # More epochs
    'grad_clip_norm': None,  # Disable gradient clipping
    'clear_cache_per_epoch': True,  # Enable cache clearing
}
```

## Next Steps

Once all checks pass:
1. ✅ Push code to GitHub/Drive
2. ✅ Open Colab
3. ✅ Run NYU Depth V2 training with real data
4. ✅ Expect similar behavior but better accuracy with real data

## Troubleshooting

### ImportError: No module named 'src'
```python
# Make sure you're running from notebooks/ directory
# The notebook adds parent directory to path
```

### CUDA Out of Memory
```python
# Reduce batch size in cell 3:
BATCH_SIZE = 8  # Reduce from 16/32
```

### MPS (Apple Silicon) Issues
```python
# Disable MPS if issues arise:
device = 'cpu'  # Force CPU
```

## Output Files

The notebook generates:
- Training curve plots (displayed inline)
- Prediction visualizations (displayed inline)
- No files saved (pure testing)

## Differences from Real Training

| Aspect | Test (Synthetic) | Real (NYU Depth V2) |
|--------|------------------|---------------------|
| Dataset | 100 train, 20 val | ~40K train, ~8K val |
| Epochs | 2 | 90 |
| Accuracy | Random (~7-15%) | 70-80% |
| Runtime | 2-3 min | 2-6 hours |
| Data | Synthetic noise | Real RGB-D scenes |

## Verification Checklist

Run the final cell to see:
```
✅ PASS - Synthetic dataset creation
✅ PASS - DataLoader creation
✅ PASS - MCResNet model instantiation
✅ PASS - Model compilation
✅ PASS - Forward pass
✅ PASS - Training with gradient clipping
✅ PASS - Training with optional cache clearing
✅ PASS - AMP enabled (if CUDA)
✅ PASS - Validation during training
✅ PASS - Model evaluation
✅ PASS - Pathway analysis
✅ PASS - Predictions
✅ PASS - Probability predictions
✅ PASS - Training curves
✅ PASS - Learning rate scheduling

🎉 ALL CHECKS PASSED!
✅ Pipeline is ready for Colab deployment
```

## Support

If you encounter issues:
1. Check cell outputs for error messages
2. Verify PyTorch installation: `torch.__version__`
3. Check device availability: `torch.cuda.is_available()`
4. Review the verification checklist output

---

**Ready to test!** Just run all cells and verify the checks pass. 🚀
