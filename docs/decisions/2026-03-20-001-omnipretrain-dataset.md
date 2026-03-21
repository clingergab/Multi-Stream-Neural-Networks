# ADR-001: OmniPretrain Dataset Design

**Date**: 2026-03-20
**Status**: Accepted
**Plan**: docs/plans/2026-03-20-omnipretrain-dataloader-plan.md

## Context

LINet pretraining requires a dataset of ~90 object categories with paired RGB and depth
tensors. The existing `SUNRGBDDataset` is designed around a fixed, pre-split directory
layout with monolithic per-split tensor files and uint8-normalized depth. The OmniPretrain
dataset has a different physical layout (per-sample `.pt` files in ImageNet-style class
folders), stores depth in uint16 millimeters, requires train/val splitting at runtime, and
needs additional depth augmentations (scale jitter, random hole dropout) that SUNRGBD does
not have. A new dataset class was introduced rather than modifying SUNRGBD.

## Decision

`OmniPretrainDataset` is a self-contained `torch.utils.data.Dataset` that walks class
folders once at `__init__`, converts depth from uint16 mm to float32 meters inline, applies
a mask-restore pattern to preserve original zero (missing) pixels through augmentation, and
delegates train/val splitting to the `get_omnipretrain_dataloaders()` factory.

## Alternatives Considered

### Alternative A: Extend SUNRGBDDataset with mode flags
- **Pros**: Single class to maintain; shared augmentation logic.
- **Cons**: SUNRGBD has fundamentally different on-disk layout and depth encoding; parameterizing both in one class produces unmaintainable conditional branches.
- **Why rejected**: The two datasets differ in layout, depth units, split strategy, and augmentation needs. A shared class would be more complex than two separate ones.

### Alternative B: NaN as depth missing-pixel sentinel
- **Pros**: Semantically clear; NaN propagates through many operations automatically.
- **Cons**: PyTorch operations (normalization, augmentation transforms) propagate NaN unpredictably; requires explicit masking at every augmentation step.
- **Why rejected**: The zero-sentinel mask-restore pattern is explicit and safe. Zero is already the raw encoding for missing depth pixels in the source data.

### Alternative C: Factory computes split; dataset class also handles split internally
- **Pros**: Simpler caller API (no `indices` parameter).
- **Cons**: If both train and val datasets compute their own splits independently, there is no guarantee they partition the data identically (especially with seeding). Coordination requires shared state.
- **Why rejected**: Passing explicit index lists from the factory to each dataset instance guarantees the split is computed once and is consistent.

## Consequences

### Positive
- Augmentation constants for depth scale jitter and hole dropout live in `augmentation_config.py` alongside all other `BASE_*` constants, maintaining the single-source-of-truth pattern.
- Zero-sentinel mask-restore is explicit: original missing pixels survive any sequence of augmentations without special-casing in each transform.
- Stratified train/val split with a rare-class fallback means datasets with very small classes do not crash; they degrade gracefully to a non-stratified split with a warning.
- `WeightedRandomSampler` handles class imbalance automatically; callers do not need to compute sample weights.

### Negative
- `_load_norm_stats` and `_WorkerInitFn` are duplicated from `sunrgbd_dataset.py`. Any future change to their behavior must be made in both places.
- The depth unit conversion (`/ 1000.0`) and the `norm_stats.json` depth statistics must be in the same units (meters). If a preprocessing script writes depth stats in a different unit, sentinel replacement silently uses the wrong fill value without a runtime error.
- sklearn is a required dependency for the stratified split; importing it at module level means the dataset cannot be used without sklearn installed.

### Neutral
- The dataset class does not perform GPU augmentation. Normalization and GPU-side augmentation follow the same `normalize` flag pattern as SUNRGBD and are the caller's responsibility.
- Class names are loaded from `class_names.txt` using a flexible parser that handles both `"bathroom"` and `"0: bathroom"` line formats, which differs from the SUNRGBD parser that only handles the indexed format.

## Related
- docs/plans/2026-03-20-omnipretrain-dataloader-plan.md — full implementation plan
- src/data_utils/sunrgbd_dataset.py — original dataset this was modeled after
- src/training/augmentation_config.py — shared augmentation constants
