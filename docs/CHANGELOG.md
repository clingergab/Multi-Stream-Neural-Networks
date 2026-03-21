# Changelog

All notable changes to this project are documented here.

---

## 2026-03-20 — OmniPretrain Dataset Loader

### Added

- `OmniPretrainDataset` — a new dataset class for loading paired RGB + depth tensor files
  from an ImageNet-style folder layout (one subfolder per class, individual `.pt` files per
  sample). Designed for LINet pretraining with ~90 object categories.

- `get_omnipretrain_dataloaders()` — factory function that returns `(train_loader,
  val_loader, num_classes)`. Handles train/val splitting, class-imbalance weighting, and
  worker seeding in one call.

- Depth uint16 mm to float32 meters conversion. Raw depth tensors stored in millimeters are
  converted to meters at load time; original zero (missing-pixel) regions are preserved
  through the full augmentation pipeline via a mask-restore pattern.

- Two new depth augmentation types, controlled by constants in `augmentation_config.py`:
  - **Depth scale jitter** — multiplies depth values by a random factor in [0.9, 1.1],
    simulating sensor calibration variation.
  - **Random hole dropout** — zeroes out 3–8 randomly placed rectangular patches of 5–20
    pixels per side, simulating sensor occlusion or missing returns.

- 90/10 stratified train/val split at runtime using sklearn. Falls back to a non-stratified
  random split with a warning when any class has fewer than 2 samples.

- `WeightedRandomSampler` on the training loader so that all classes are sampled at equal
  frequency by default, regardless of class size.

- New constants in `src/training/augmentation_config.py`:
  `BASE_DEPTH_SCALE_JITTER_P`, `BASE_DEPTH_SCALE_MIN`, `BASE_DEPTH_SCALE_MAX`,
  `BASE_HOLE_DROPOUT_P`, `BASE_HOLE_DROPOUT_NUM_MIN`, `BASE_HOLE_DROPOUT_NUM_MAX`,
  `BASE_HOLE_DROPOUT_SIZE_MIN`, `BASE_HOLE_DROPOUT_SIZE_MAX`.

- 43 unit tests covering dataset construction, depth unit conversion, zero-sentinel
  preservation, augmentation constants, train/val split consistency, and the factory
  function API.

### Changed

- `src/data_utils/__init__.py` now exports `OmniPretrainDataset` and
  `get_omnipretrain_dataloaders`.

### Notes

- `norm_stats.json` for OmniPretrain datasets **must** contain depth statistics computed in
  meters (divide raw uint16 values by 1000.0 before computing mean/std). If the
  preprocessing script computes stats in a different unit, missing-pixel replacement will
  use the wrong fill value without a runtime error.
- The augmentation pipeline matches SUNRGBD's CPU-side transform structure, making it
  compatible with the same weight-transfer workflow.
