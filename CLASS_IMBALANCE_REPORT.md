# NYU Depth V2 Class Imbalance Report

## 🚨 CRITICAL ISSUE: Severe Class Imbalance

### Summary
The NYU Depth V2 dataset has a **192x class imbalance ratio** - one of the most severe imbalances possible. This **will cause training to fail** without proper handling.

---

## 📊 The Numbers

### Class Distribution
- **Total samples:** 1,449
- **Number of classes:** 27
- **Imbalance ratio:** 192:1 (bedroom vs indoor_balcony)

### Dominant Classes (73.6% of dataset)
| Class | Name | Samples | % of Dataset |
|-------|------|---------|--------------|
| 2 | Bedroom | 383 | 26.4% |
| 16 | Kitchen | 225 | 15.5% |
| 18 | Living Room | 221 | 15.3% |
| 1 | Bathroom | 121 | 8.4% |
| 9 | Dining Room | 117 | 8.1% |

### Ultra-Rare Classes (≤4 samples - UNLEARNABLE)
| Class | Name | Samples |
|-------|------|---------|
| 15 | Indoor Balcony | 2 |
| 10 | Exercise Room | 3 |
| 17 | Laundry Room | 3 |
| 22 | Printer Room | 3 |
| 8 | Dinette | 4 |
| 11 | Foyer | 4 |

### Effective Information
- **Entropy:** 71.7% of maximum
- **Effective classes:** ~10.6 out of 27
- **Top 3 classes:** 57.2% of data
- **Top 10 classes:** 90.5% of data

---

## ⚠️ Impact on Training

### What This Caused in Your Training:

1. **Model bias toward common classes**
   - Always predicts bedroom/kitchen/living room
   - Gets ~26% accuracy by just predicting bedroom
   - Validation stuck at 14% (predicting top classes)

2. **Rare classes never learned**
   - Classes with 2-4 samples impossible to learn
   - Model assigns them 0% probability
   - Stream2 (depth) collapsed because it couldn't distinguish rare room types

3. **False sense of learning**
   - Training acc 58% looks good
   - But it's just memorizing "bedroom" for most inputs
   - Validation 14% is actually terrible for 27 classes

4. **Stream imbalance**
   - RGB can memorize common textures (beds, kitchens)
   - Depth struggles with rare geometric layouts
   - This is why RGB overfitted more

---

## ✅ Solutions (REQUIRED)

### Solution 1: Class Weights (IMPLEMENTED)
**Best for:** Keeping all 27 classes

```python
from class_weights_nyu import create_weighted_loss

# Instead of:
# criterion = nn.CrossEntropyLoss()

# Use:
criterion = create_weighted_loss(device='cuda')
```

**How it works:**
- Rare classes get higher loss weight (3.7x for indoor_balcony)
- Common classes get lower loss weight (0.02x for bedroom)
- Forces model to learn all classes, not just common ones

### Solution 2: Remove Ultra-Rare Classes
**Best for:** More stable training

```python
# Remove 10 classes with <5 samples
# Reduces to 17 learnable classes
# Update num_classes=17 in model
```

### Solution 3: Focal Loss
**Best for:** Hard examples

```python
# pip install focal-loss-torch
from focal_loss import FocalLoss

criterion = FocalLoss(
    alpha=NYU_CLASS_WEIGHTS,  # class weights
    gamma=2.0,  # focus on hard examples
    reduction='mean'
)
```

### Solution 4: Stratified Sampling (Already Implemented)
**What we did:**
- Added shuffling before train/val split
- Ensures val has 22/27 classes (was only 3!)
- Both train and val see diverse samples

---

## 📋 Recommendation for Your Training

### Immediate Actions:
1. ✅ **Add class weights** (highest priority)
   ```python
   from class_weights_nyu import create_weighted_loss
   criterion = create_weighted_loss(device='cuda')
   ```

2. ✅ **Monitor per-class accuracy**
   - Track accuracy for top 5 and bottom 5 classes
   - Ensure rare classes aren't stuck at 0%

3. ⚠️ **Consider removing ultra-rare classes**
   - 10 classes have <5 samples (unlearnable)
   - Option: Merge them into "other" category
   - Or: Remove entirely, use 17 classes

### Expected Improvements:
- **Validation accuracy:** Should reach 25-35% (instead of 14%)
- **Stream balance:** Depth stream won't collapse
- **Rare class learning:** Will get >0% accuracy on rare rooms

### Long-term:
- Collect more data for rare classes
- Or: Focus only on common room types (bedroom, kitchen, living room, bathroom)

---

## 🔬 Technical Details

### Class Weights Formula:
```python
weight[i] = total_samples / (num_classes * class_count[i])
weights = weights / weights.sum() * num_classes  # normalize
```

### Full Weight Vector:
```python
NYU_CLASS_WEIGHTS = [
    1.0626,  # basement
    0.0615,  # bathroom
    0.0194,  # bedroom (most common → lowest weight)
    0.2066,  # bookstore
    1.4876,  # cafe
    0.1518,  # classroom
    1.2397,  # computer_lab
    1.4876,  # conference_room
    1.8596,  # dinette
    0.0636,  # dining_room
    2.4794,  # excercise_room
    1.8596,  # foyer
    0.2755,  # furniture_store
    0.1488,  # home_office
    1.4876,  # home_storage
    3.7191,  # indoor_balcony (most rare → highest weight)
    0.0331,  # kitchen
    2.4794,  # laundry_room
    0.0337,  # living_room
    0.0954,  # office
    0.7438,  # office_kitchen
    0.2399,  # playroom
    2.4794,  # printer_room
    0.4375,  # reception_room
    1.4876,  # student_lounge
    0.2975,  # study
    1.0626,  # study_room
]
```

---

## 🎯 Success Criteria

### Before Class Weights:
- Val accuracy: ~14% (stuck)
- Bedroom prediction rate: ~26% (biased)
- Rare class accuracy: 0%

### After Class Weights (Expected):
- Val accuracy: 25-35%
- Balanced predictions across classes
- Rare classes: >5% accuracy each
- Stream2 no longer collapses

---

## Files Created

1. `class_weights_nyu.py` - Pre-computed class weights
2. `test_class_imbalance.py` - Full analysis script
3. `CLASS_IMBALANCE_REPORT.md` - This report

## Next Steps

1. **Update your Colab notebook** to use weighted loss
2. **Restart training** from scratch
3. **Monitor per-class metrics** during training
4. **Report results** after 5-10 epochs
