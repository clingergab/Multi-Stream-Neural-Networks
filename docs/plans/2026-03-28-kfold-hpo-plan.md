# Development Plan: K-Fold Cross-Validation with Live Reporting for Ray Tune HPO

## 1. Objective & Scope

**What we are building:** Modifying five cells in the notebook `/Users/gclinger/Documents/projects/Multi-Stream-Neural-Networks/notebooks/colab_LiNet3_SUN_hype_tune_kfold.ipynb` to implement proper k-fold cross-validation within each Ray Tune trial, with live per-epoch metric reporting. Each trial runs 5 stratified folds to completion, and HyperOptSearch optimizes on the final mean composite metric.

**Why:** The current implementation has a partially-applied first attempt with broken ASHA scheduler configuration (`max_t=N_FOLDS, grace_period=2` -- ASHA would prematurely kill trials mid-fold). The user wants every trial to run all 5 folds, with HyperOptSearch selecting configs based on completed trial results only.

**Out of scope:**
- Changes to `src/` modules (li_net.py, sunrgbd_dataset.py, etc.) -- all changes are notebook-only
- Changes to the search space or hyperparameter ranges
- Changes to analysis cells (20-22) -- already updated for new metric names
- Reducing `max_concurrent_trials` -- keep at 10, user can adjust
- Changes to the configuration cell (cell 17, id `6107f3d4`) -- already correct

## 2. Architecture & Design Decisions

### High-level approach

Each Ray Tune trial calls `train_linet_tune(config)`, which:
1. Reconstructs a single `SUNRGBDDataset` for the train split from `data_root` (passed via `tune.with_parameters`). This is the existing pattern and avoids Ray serialization issues with mmap tensors. The datasets use mmap so the OS shares physical pages across trials anyway.
2. Runs `StratifiedKFold(n_splits=5)` over it
3. For each fold: creates train/val loaders from fold indices, builds a fresh model, trains with `model.fit()`, collects best metrics via a callback
4. Reports metrics to Ray Tune both **per-epoch within each fold** (for live visibility) and **after each fold completes** (running cross-fold means)
5. The final `tune.report()` after fold 5 carries the overall mean/std -- this is what HyperOptSearch uses

### Key design decisions

**Decision 1: Single callback class (`KFoldTuneReporter`) that both tracks per-fold bests AND calls `tune.report()`.**

Rationale: The current code splits this into `FoldMetricsTracker` (no reporting) and external `tune.report()` calls. Merging them into one callback gives per-epoch live reporting. The callback receives fold index and a mutable list of completed-fold results at construction, so it can compute running cross-fold means.

Rejected alternative: Keep `FoldMetricsTracker` separate and report only between folds. This loses per-epoch visibility, which the user explicitly requested.

**Decision 2: No ASHA scheduler -- omit `scheduler=` from TuneConfig.**

Rationale: ASHA's `training_iteration` would correspond to individual `tune.report()` calls (hundreds per trial). With no scheduler, every trial runs to completion. HyperOptSearch is a search algorithm, not a scheduler -- it works independently.

Rejected alternative: ASHA with `max_t=N_FOLDS` and reporting only per-fold. This defeats the purpose of running all folds to completion, and per-epoch visibility would be lost.

**Decision 3: Fresh model per fold (no weight sharing across folds).**

Rationale: Standard k-fold practice. Each fold is independent. The composite metric is the mean across folds, giving a more robust estimate of generalization.

**Decision 4: Composite metric computed from running means (not per-fold composites averaged).**

The composite formula `val_mca - 5 * (gap^3)` is nonlinear, so `mean(composite_per_fold)` differs from `composite(mean_val_mca, mean_train_mca)`. We compute composite from the running mean val_mca and mean train_mca. This is more meaningful: it penalizes the overall gap, not individual fold gaps.

Rejected alternative: Average per-fold composites. The cubic penalty would over-penalize folds where one fold happens to have a large gap but the mean gap is small.

## 3. Implementation Details

### Cells to modify

#### Cell id `5694e239` (setup-imports cell) -- Remove ASHA import, add StratifiedKFold

Remove the line `from ray.tune.schedulers import ASHAScheduler`. Add `from sklearn.model_selection import StratifiedKFold` (grouped with third-party imports). All other imports in the cell remain unchanged.

#### Cell id `5dbc3f03` (markdown section header for 8b) -- Update description

Update the bullet points: remove "ASHA Scheduler" references, add "K-Fold CV: 5-fold stratified cross-validation per trial, all folds run to completion". Keep remaining bullets about parallel trials and fit() method.

#### Cell id `5f0bfd6f` (training function cell) -- Replace entire cell

New contents:

1. **`N_FOLDS = 5`** -- module-level constant

2. **`class KFoldTuneReporter`** -- callback passed to `model.fit()` per fold:
   - Constructor: `(fold_idx, n_folds, completed_fold_results: list[dict])`
   - `on_epoch_end(epoch, logs)`: tracks fold's best val_mca; on improvement, snapshots train_mca, val_acc (from `logs['val_accuracy']`), and val_loss (from `logs['val_loss']`) at that epoch; computes running cross-fold means from completed folds + current fold's best-so-far; calls `tune.report()`
   - `get_fold_result() -> dict`: returns `{best_val_mca, best_train_mca, best_val_acc, best_val_loss}`

3. **`def _build_fold_loaders(...)`** -- creates DataLoaders for one fold with WeightedRandomSampler. Class weights are computed from `[dataset.labels[i] for i in train_indices]`, not the full dataset.

4. **`def _build_and_train_fold(...)`** -- builds fresh model/optimizer/scheduler, calls model.fit() with KFoldTuneReporter, returns fold result

5. **`def train_linet_tune(config, ...)`** -- the Ray Tune trainable: reconstructs datasets from `data_root` (passed via `tune.with_parameters`, avoiding Ray serialization issues with mmap tensors), loops over folds, accumulates results, final tune.report() with mean/std/per-fold

#### Cell id `4ab63894` (Ray Tune config/run cell) -- Update scheduler/reporter

- Remove ASHA scheduler creation and `scheduler=` from TuneConfig
- `HyperOptSearch(metric="mean_composite", mode="max", n_initial_points=15)`
- `BestTrialReporter(metric="mean_val_mca", mode="max", every_n_results=20)`
- Update CLIReporter metric_columns:
  ```python
  {
      "fold": "fold",
      "cur_best_val_mca": "fold_mca",
      "mean_val_mca": "mean_mca",
      "mean_composite": "composite",
      "n_folds_completed": "folds",
  }
  ```
- `best_result = results.get_best_result("mean_composite", "max")`
- Update post-tuning prints for new metric names

## 4. Interface Contracts

### `KFoldTuneReporter`

```python
class KFoldTuneReporter:
    def __init__(
        self,
        fold_idx: int,               # 0-based fold index
        n_folds: int,                 # total number of folds (5)
        completed_fold_results: list[dict],  # mutable list, grows as folds complete
    ) -> None:
        """Initializes per-fold best tracking (val_mca, train_mca, val_acc, val_loss)."""

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """Called by model.fit() after each epoch.

        logs keys: train_loss, train_accuracy, val_loss, val_accuracy, train_mca, val_mca

        Behavior:
        - Initialize best_val_mca to -1.0 (not 0.0) so the first epoch always
          triggers a snapshot, even if val_mca is 0.0 for a degenerate config.
        - If logs['val_mca'] > current fold's best val_mca: update best_val_mca
          AND snapshot best_train_mca (from logs['train_mca']),
          best_val_acc (from logs['val_accuracy']),
          best_val_loss (from logs['val_loss']) at this epoch.
        - Compute running means over completed_fold_results + current fold's
          best-so-far for: val_mca, train_mca, val_acc, val_loss, gap, composite.
        - Call tune.report() with the dict described below.
        """

    def get_fold_result(self) -> dict:
        """Returns {'best_val_mca': float, 'best_train_mca': float,
                    'best_val_acc': float, 'best_val_loss': float}"""
```

### `_build_fold_loaders`

```python
def _build_fold_loaders(
    train_dataset: SUNRGBDDataset,   # full training dataset (augmentation ON)
    val_dataset: SUNRGBDDataset,     # same data, split='val' (augmentation OFF)
    train_indices: list[int],        # indices for this fold's training split
    val_indices: list[int],          # indices for this fold's validation split
    batch_size: int,                 # from config, default 64
    seed: int,                       # for WeightedRandomSampler generator
    fold_idx: int,                   # for worker_init_fn seed offset
) -> tuple[DataLoader, DataLoader]:
    """Creates train and val DataLoaders for one fold.

    Train loader uses WeightedRandomSampler for class balancing.
    Class weights are computed from [dataset.labels[i] for i in train_indices],
    NOT from the full dataset labels. Each class weight = num_fold_train_samples / class_count.

    Val loader uses sequential sampling (no shuffle, no weighting).
    Both loaders use num_workers=1, prefetch_factor=2, pin_memory=True.
    Train loader: persistent_workers=True. Val loader: persistent_workers=False.
    """
```

### `_build_and_train_fold`

```python
def _build_and_train_fold(
    config: dict,                          # Ray Tune hyperparameter config
    train_dataset: SUNRGBDDataset,         # full training dataset
    val_dataset: SUNRGBDDataset,           # val-mode dataset
    train_indices: list[int],              # fold train indices
    val_indices: list[int],                # fold val indices
    fold_idx: int,                         # 0-based fold index
    n_folds: int,                          # total folds
    completed_fold_results: list[dict],    # mutable accumulator
    norm_stats: dict,                      # normalization statistics
    pretrained_weights_path: str | None,   # optional pretrained weights
    seed: int,                             # random seed (offset by fold_idx)
) -> dict:
    """Builds a fresh model, trains it on one fold, returns fold result dict.

    Steps:
    1. Call _build_fold_loaders to get train/val loaders
    2. Create fresh li_resnet18 model with config hyperparameters
    3. Optionally load pretrained weights (skip fc if num_classes mismatch)
    4. Create optimizer, scheduler, compile model
    5. Call model.fit(verbose=False, early_stopping=True, patience=15,
       monitor='val_mca') with KFoldTuneReporter callback
    6. After fit: del model, train_loader, val_loader; torch.cuda.empty_cache()
    7. Return reporter.get_fold_result()

    Each fold uses seed + fold_idx for model init seeding and worker init seeding,
    ensuring different folds get different random sequences.
    std_val_mca uses sample standard deviation (ddof=1).
    """
```

### `train_linet_tune`

```python
def train_linet_tune(
    config: dict,                          # Ray Tune hyperparameter config
    *,
    data_root: str = None,                 # dataset path, passed via tune.with_parameters
    norm_stats: dict = None,               # normalization stats, passed via tune.with_parameters
    pretrained_weights_path: str | None = None,  # optional, via tune.with_parameters
    seed: int = 42,                        # via tune.with_parameters
) -> None:
    """Top-level Ray Tune trainable. Reconstructs datasets from data_root each trial
    (avoids Ray serialization issues with mmap tensors; OS shares pages anyway).

    Steps:
    1. set_seed(seed, deterministic=False)
    2. Create SUNRGBDDataset(data_root=data_root, split='train', normalize=False, **aug_config)
    3. Create val_dataset: SUNRGBDDataset(data_root=data_root, split='train', normalize=False)
       then monkey-patch val_dataset.split = 'val' to disable augmentation in __getitem__
       WITHOUT changing the data directory. (No val/ directory exists -- preprocessed with --no-val-split.)
    4. StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    5. Loop over folds: call _build_and_train_fold, append result to completed_fold_results
    6. After all folds: compute final means/stds, final tune.report() with done=True

    batch_size defaults to config.get('batch_size', 64).
    t_max defaults to config.get('t_max', 110).
    """
```

### `on_epoch_end` reported dict (intermediate, per-epoch)

```python
{
    # Current fold info
    'fold': int,                    # 1-based fold number
    'epoch': int,                   # 0-based epoch within fold
    'cur_val_mca': float,           # this epoch's raw val_mca
    'cur_train_mca': float,         # this epoch's raw train_mca
    'cur_best_val_mca': float,      # current fold's best val_mca so far
    'cur_best_train_mca': float,    # train_mca at epoch of best val_mca

    # Running cross-fold means (completed folds + current fold's best)
    'mean_val_mca': float,
    'mean_train_mca': float,
    'mean_val_acc': float,
    'mean_val_loss': float,
    'mean_gap': float,
    'mean_composite': float,
    'n_folds_completed': int,       # number of fully completed folds

    # Per-fold breakdown -- val_mca only for completed folds
    'fold_0_val_mca': float,        # present after fold 0 completes
    'fold_1_val_mca': float,        # present after fold 1 completes
    # ...
}
```

### Final `tune.report()` (after all 5 folds)

The final report includes additional per-fold `val_acc` keys and `std_val_mca` that are NOT present in intermediate per-epoch reports. This asymmetry is intentional: intermediate reports prioritize compactness and the most important metric (val_mca per fold), while the final report provides the full breakdown needed for analysis.

```python
{
    'mean_val_mca': float,
    'std_val_mca': float,           # only in final report
    'mean_train_mca': float,
    'mean_val_acc': float,
    'mean_val_loss': float,
    'mean_gap': float,
    'mean_composite': float,
    'n_folds_completed': 5,
    'fold_0_val_mca': float,
    'fold_1_val_mca': float,
    'fold_2_val_mca': float,
    'fold_3_val_mca': float,
    'fold_4_val_mca': float,
    'fold_0_val_acc': float,        # only in final report
    'fold_1_val_acc': float,        # only in final report
    'fold_2_val_acc': float,        # only in final report
    'fold_3_val_acc': float,        # only in final report
    'fold_4_val_acc': float,        # only in final report
    'all_folds_complete': True,
}
```

## 5. Testing Strategy

Notebook-only change -- no unit tests. Verification via manual checklist:

- [ ] No `ASHAScheduler` in imports or TuneConfig
- [ ] Markdown section header no longer mentions ASHA
- [ ] Each trial reports live per-epoch metrics (visible in CLIReporter)
- [ ] Final report per trial has `mean_composite`, `mean_val_mca`, `std_val_mca`
- [ ] CLIReporter shows `fold`, `fold_mca`, `mean_mca`, `composite` columns
- [ ] BestTrialReporter tracks best `mean_val_mca` across all trials
- [ ] CUDA memory freed between folds (no OOM)
- [ ] `batch_size` and `t_max` default to 64 and 110 when not in search space
- [ ] `val_mca` in first epoch's `tune.report` is non-zero (confirms val_loader was passed and evaluated)
- [ ] `std_val_mca` in best result is > 0 (confirms all 5 folds ran and produced different results)
- [ ] WeightedRandomSampler uses fold-local label counts (from `[dataset.labels[i] for i in train_indices]`), not full-dataset counts

## 6. Risks & Mitigations

1. **GPU memory**: 10 concurrent trials x models. Mitigated by `del model; torch.cuda.empty_cache()` between folds.
2. **5x longer trials**: User's explicit design choice. HyperOptSearch Bayesian optimization + per-fold early stopping help.
3. **High tune.report() volume (~550 calls per trial)**: Within Ray Tune's design. `CLIReporter(max_report_frequency=30)` throttles how often the progress table is reprinted to console (CLIReporter constructor parameter).
4. **Interrupted trial restarts from fold 0**: Acceptable -- completed trials preserved via Drive sync.

## Review History

### Round 1
- **Reviewer**: plan-reviewer
- **Status**: NEEDS REVISION
- **Issues**: Missing cells (imports, markdown), incomplete interface contracts, dataset reconstruction unclear, testing gaps, Risk 3 terminology
- **Resolution**: All 6 issues addressed in revision. Added 2 cells to modify, full type signatures for all functions, explicit dataset reconstruction pattern, 3 new test items, corrected Risk 3.

### Gap Analysis Round 1
- **Auditor**: gap-finder
- **Status**: ISSUES FOUND (2 critical, 4 important, 4 minor)
- **Critical fixes applied**:
  1. Val dataset creation: explicit monkey-patch pattern documented (`split='train'` then `.split = 'val'`)
  2. `KFoldTuneReporter` best_val_mca initialized to -1.0 (not 0.0) to ensure first epoch triggers snapshot
- **Important fixes applied**:
  3. Added `from sklearn.model_selection import StratifiedKFold` to imports cell changes
  4. Added `del train_loader, val_loader` to cleanup between folds
  5. Specified `verbose=False` for `model.fit()` calls
  6. Specified `early_stopping=True, patience=15, monitor='val_mca'` for `model.fit()` calls
- **Minor fixes applied**:
  7. Replaced `'done': True` with `'all_folds_complete': True` (avoid Ray reserved key)
  8. Changed CLIReporter `"done"` column to `"folds"` for clarity
  9. Specified `seed + fold_idx` for per-fold seeding
  10. Specified sample std (ddof=1) for `std_val_mca`
