# Changelog

All notable changes to this project are documented here.

---

## 2026-04-24 — Pluggable Classifier Head (Linear + Max-Out)

### Added

- `LinearHead` and `MaxoutHead` — two pluggable `nn.Module` classifier heads in a new file
  `src/models/linear_integration/li_net3/heads.py`. `LinearHead` is a thin wrapper around
  `nn.Linear`. `MaxoutHead` projects features to `C·K` logits, reshapes to
  `(B, num_classes, num_subnodes)`, and takes `.amax(dim=2)`, giving each class K parallel
  sub-classifiers whose maximum score is used for prediction. Both return `(B, num_classes)`
  logits; no downstream changes were required.

- `classifier_head: str = 'linear'` and `num_subnodes: int = 3` kwargs on `LINet.__init__`
  and `li_resnet18`. The default `'linear'` preserves byte-identical behavior relative to the
  pre-refactor baseline (verified by `torch.equal` on weights and forward output, no weight
  copying). An unknown head name raises `ValueError` at construction time.

- Sub-node utilization diagnostic cell (cell 69, `# --- 19.12 ---`) in
  `notebooks/colab_LINet3_SUN_training_with_val.ipynb`. After training, a forward hook on
  `model.fc.fc` captures pre-max logits, computes per-class sub-node utilization counts and
  normalized entropy, renders a bar chart, and saves results to
  `results/maxout_subnode_utilization_{timestamp}.json` with a full config block. The cell
  skips gracefully if the model's head is `'linear'`.

- Hard-abort in the notebook's pretrained-load block (cell 21): `load_state_dict(strict=False)`
  return value is now checked explicitly. If any keys beyond the expected classifier-head
  key swap (`fc.weight`/`fc.bias` → `fc.fc.weight`/`fc.fc.bias`) appear in the mismatch
  sets, a `RuntimeError` is raised immediately rather than letting the run continue silently
  with a corrupt backbone.

- 8 new unit tests in `tests/test_classifier_head.py` covering: output shapes, the K=0
  rejection assertion, head-level math equivalence at K=1 (with explicit weight copying),
  RNG-consumption equivalence of the full model at K=1 (no weight copying — tests that
  construction order is identical), `ValueError` on unknown head name, and a slow behavioral
  test (`test_maxout_k3_argmax_specialization`) verifying that K=3 sub-nodes specialize on
  XOR-like 2-mode-per-class synthetic data (mean cross-cluster disagreement ≥ 0.7).

- `LinearHead` and `MaxoutHead` added to the `src/models/linear_integration/li_net3`
  package exports.

### Changed

- `MODEL_CONFIG` in notebook cell 16 now includes `'classifier_head': 'maxout'` and
  `'num_subnodes': 3`. The notebook default is `'maxout'`; the LINet class default remains
  `'linear'` for backward compatibility.

- Notebook cell 21 prints `Classifier head`, `Sub-nodes per class`, and
  `Classifier head params` after model construction, and passes both new kwargs through to
  `li_resnet18`.

- `tests/diagnose_gradient_diff.py`, `tests/test_conv1_gradient_flow.py`, and
  `tests/src/models/abstracts/test_param_groups.py` updated to access the classifier weight
  at `model.fc.fc.weight` (the inner `nn.Linear` of the head module) rather than
  `model.fc.weight` (which is no longer valid).

### Notes

- The state-dict key for the classifier changed from `fc.weight`/`fc.bias` to
  `fc.fc.weight`/`fc.fc.bias`. Pre-refactor checkpoints loaded with `strict=False` will
  silently skip those two keys; all backbone keys still load correctly. The notebook's
  hard-abort guard catches any mismatch beyond this expected swap.
- Gates 4 (K=1 end-to-end within ±0.1pp of baseline) and 5 (K=3 full training run;
  primary signal: test `dining_area` accuracy above the 25% baseline) require a GPU machine
  and the SUN RGB-D dataset. Infrastructure is in place; experimental results will be
  documented separately.

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
