# ScanNet Pretraining Pipeline -- Development Plan

## 1. Objective & Scope

### What we are building

A pipeline to use ScanNet as pretraining data for the Multi-Stream Neural Network (LINet) model:

1. **Preprocessing notebook** (`notebooks/scannet_preprocess.ipynb`) -- downloads ScanNet `.sens` + `.txt` files, extracts evenly-spaced frames, center-crops to 256x256, saves as paired `_rgb.pt`/`_depth.pt` tensors, computes normalization stats, and packages for Google Drive upload.
2. **Dataloader module** (`src/data_utils/scannet_pretrain_dataset.py`) -- loads the preprocessed tensors with the same interface as `OmniPretrainDataset` (same augmentation pipeline, same return signature).

The new dataloader must be a drop-in replacement for `get_omnipretrain_dataloaders` in the existing training notebooks so the training loop requires zero changes.

### What is OUT of scope

- Modifying the existing `omnipretrain_dataset.py` or `omni_preprocess.ipynb` (they remain for OmniObject3D)
- Downloading `.ply` meshes or 2D semantic labels from ScanNet
- Object-level semantic segmentation or instance segmentation tasks
- Modifications to the LINet model architecture
- Creating new training notebooks (existing Omni training notebooks will be updated separately)
- GPU-based augmentation path (`gpu_augmentation.py`) changes for ScanNet

## 2. Architecture & Design Decisions

### High-level approach

Create new files (`scannet_pretrain_dataset.py` and `scannet_preprocess.ipynb`) modeled closely on the Omni equivalents, adapting for ScanNet's data format (binary `.sens` files parsed by ScanNet's official `SensorData.py`, scene-type labels from `.txt` metadata).

### Key design decisions

**Decision 1: New files, not in-place modification of Omni files.**
- Rationale: OmniObject3D and ScanNet are fundamentally different data sources (synthetic object renders vs. real indoor scans). Keeping them separate avoids merge conflicts, allows independent iteration, and lets users run either pipeline.
- The helper functions (`_load_class_names`, `_load_norm_stats`, `_discover_samples`, `_WorkerInitFn`) are **copied** from `omnipretrain_dataset.py`, not imported. This keeps the modules fully decoupled.
- Rejected alternative: Modifying `omnipretrain_dataset.py` in-place with a `dataset_type` flag.

**Decision 2: Scene-level labels from `sceneType` field in ScanNet `.txt` metadata files.**
- Rationale: ScanNet's 20-class NYU40 benchmark is object-level (per-point). For scene recognition pretraining, we need scene-level labels. Each ScanNet scene has a `sceneType` string in its `<scene_id>.txt` metadata (e.g., "Bathroom", "Office", "Kitchen"). We define a canonical mapping from these strings to integer class indices.
- The `.txt` files use `key = value` format (split on first `=`, strip whitespace from key and value). The `sceneType` field is always present.
- Rejected alternative: Using ScanNet's 20 NYU40 object classes. These are per-point labels, not scene-level.

**Decision 3: Use only the 20 labeled scene types, no catch-all.**
- Rationale: ScanNet defines 21 scene types in `scene_types_all.txt` (IDs 1-21). The 21st is "Misc" — scenes that don't fit any category. We exclude Misc scenes entirely and keep only the 20 meaningful types. Scenes whose sceneType doesn't match any of the 20 are skipped during preprocessing.
- The 20 scene types (from ScanNet's official list):
  1. Apartment, 2. Bathroom, 3. Bedroom/Hotel, 4. Bookstore/Library,
  5. Classroom, 6. Closet, 7. Computer Cluster, 8. Conference Room,
  9. Copy/Mail Room, 10. Dining Room, 11. Game Room, 12. Gym,
  13. Hallway, 14. Kitchen, 15. Laundry Room, 16. Living Room/Lounge,
  17. Lobby, 18. Office, 19. Stairs, 20. Storage/Basement/Garage
- `SCENE_TYPE_ALIASES` is built from actual data: Cell 14 prints all unique raw `sceneType` strings, and the alias dict is populated based on those results (e.g., `'livingroom' -> 'living_room'`, `'Living room' -> 'living_room'`). Aliases live only in the notebook — the dataloader never sees raw scene type strings, only integer labels.
- Rejected alternative: Including a catch-all "other" class.

**Decision 4: Use ScanNet's official train/val split directly (train=1,201 scenes for training, val=312 scenes for validation).**
- Rationale: Since ScanNet is used purely for pretraining (not benchmark evaluation), we don't need a held-out test set. The official v2 split gives us train (1,201 scenes) and val (312 scenes). The 100 test scenes are excluded (hidden labels).
- The notebook uses `scannetv2_train.txt` and `scannetv2_val.txt` from the ScanNet benchmark repo.
- **The micro-loop iterates ONLY over scene IDs in these two files.** Test scenes are never downloaded or processed, even though their `.txt` metadata may exist on disk.
- All frames from one scene go to the same split (scene-level split prevents data leakage).

**Decision 5: Frame sampling -- target 66 evenly-spaced frames per scene, with min-frame threshold.**
- Rationale: 66 frames across ~1,513 scenes yields ~100,000 frames. Evenly spacing maximizes spatial diversity within a scene. Research confirms ScanNet scenes have ~800-2,500 frames (avg ~1,652 at 30fps). With 66 samples from a 1,652-frame scene, that's one frame every ~25 frames (~0.8 seconds of camera motion) — producing visually distinct views. Scenes with fewer than 66 frames use all available frames. Scenes with fewer than 10 frames are skipped (likely corrupted or too short).
- Verification: The notebook logs per-scene frame counts, sampled indices, and checks for near-duplicate frames (via simple pixel-difference threshold on a sample of scenes). Note: pixel-difference is a heuristic, not definitive — it may flag uniform walls as duplicates. The even-spacing heuristic is the primary guarantee of diversity.
- Rejected alternative: Fixed frame-rate sampling (e.g., every Nth frame).

**Decision 6: Depth stored as uint16 millimeters (same as Omni convention).**
- Rationale: ScanNet's `.sens` files store depth as compressed uint16 values. The header contains a `depth_shift` field (float32 divisor, typically 1000.0) that converts raw values to meters: `depth_meters = raw_uint16 / depth_shift`. With `depth_shift=1000`, the raw uint16 values ARE millimeters — which is exactly our storage convention.
- **No transformation needed during preprocessing.** `SensorData.py` extracts the raw uint16 depth arrays without applying `depth_shift`. Since these are already millimeters, we save them directly as uint16 `[1, H, W]` tensors. The `__getitem__` method handles the `/ 1000.0` conversion to meters at load time, matching the existing Omni pipeline.
- The extraction code should assert `depth_shift == 1000.0` as a sanity check and log a warning if any scene has a different value (none are expected to).
- Rejected alternative: Storing as float32 meters (4x storage).

**Decision 7: Synchronized random erasing between RGB and depth.**
- Rationale: The Omni dataloader applies independent random erasing boxes to RGB and depth (separate probability rolls, separate box coordinates). This creates spatially impossible occlusion patterns that can harm fusion layers. The ScanNet dataloader fixes this.
- Implementation: Remove the two separate `v2.RandomErasing` transform instances. Instead, in `__getitem__`, when erasing is triggered:
  1. Sample erase parameters once: `i, j, h, w, value = v2.RandomErasing.get_params(tensor, scale=..., ratio=..., value=0)`
  2. Apply `F2.erase(rgb, i, j, h, w, value)` and `F2.erase(depth, i, j, h, w, value)` with the same coordinates.
  3. RGB and depth still have independent probability rolls for WHETHER to erase, but when both fire, they share the same box. When only one fires, it erases independently.

**Decision 8: Fix val_loader `num_workers` bug from Omni.**
- The Omni dataloader has a latent bug: when `num_workers=1`, `val_workers = num_workers // 2 = 0`, but `persistent_workers` and `prefetch_factor` checks use the original `num_workers` (which is > 0), causing a crash.
- The ScanNet dataloader fixes this: compute `val_workers = max(num_workers // 2, 0)`, then use `val_workers` for both the DataLoader `num_workers` and the `persistent_workers`/`prefetch_factor` guards.

### Architecture overview

```
notebooks/scannet_preprocess.ipynb    (Colab notebook -- preprocessing)
    |
    v
data/scannet_pretrain_256/            (output directory)
    class_names.txt
    norm_stats.json
    dataset_info.txt
    train/<scene_type>/<scene_id>_f<frame_idx>_rgb.pt
    train/<scene_type>/<scene_id>_f<frame_idx>_depth.pt
    val/<scene_type>/...
    |
    v
src/data_utils/scannet_pretrain_dataset.py   (dataloader)
    |
    v
Existing training loop (unchanged)
```

### Disk budget (Colab)

- Each RGB tensor: uint8 `[3, 256, 256]` = ~192KB + .pt overhead ≈ ~200KB
- Each depth tensor: uint16 `[1, 256, 256]` = ~128KB + .pt overhead ≈ ~140KB
- Per frame pair: ~340KB
- 100,000 frames: ~34GB
- Final `tar.gz` (compressed): ~20-25GB
- Colab disk (A100): ~200GB usable
- Peak usage: ~34GB (tensors) + ~800MB (4 concurrent .sens files) + tar.gz = ~60GB
- Headroom: sufficient, but the notebook includes a disk check before starting.

## 3. Implementation Details

### File 1: `src/data_utils/scannet_pretrain_dataset.py` (NEW)

**Purpose**: Dataset class and dataloader factory for ScanNet pretraining data.

**Contents** (all functions copied from `omnipretrain_dataset.py`, NOT imported):
- `_load_class_names(data_root: str) -> list[str]`: Loads from `class_names.txt`.
- `_load_norm_stats(data_root: str) -> dict`: Loads from `norm_stats.json`.
- `_discover_samples(data_root: str, class_names: list[str]) -> list[tuple[str, str, int]]`: Walks class folders, finds paired `_rgb.pt`/`_depth.pt`.
- `ScanNetPretrainDataset(Dataset)`: Same augmentation pipeline as `OmniPretrainDataset` except for random erasing, which is **synchronized between modalities** (see Decision 7). The `_compute_scaled_aug_values` method creates a single `_erasing_transform` for parameter sampling, and `__getitem__` uses `F2.erase()` with shared coordinates.
- `_WorkerInitFn`: Picklable worker init (seeds numpy, random, torch).
- `get_scannet_pretrain_dataloaders(...)`: Factory function with same return signature as `get_omnipretrain_dataloaders`. Includes the val_loader `num_workers` bug fix (Decision 8).

### File 2: `src/data_utils/__init__.py` (MODIFIED)

**Purpose**: Register the new ScanNet dataset exports alongside OmniPretrain.

**Changes**: Add imports of `ScanNetPretrainDataset` and `get_scannet_pretrain_dataloaders` from the new module. Add to `__all__`.

### File 3: `notebooks/scannet_preprocess.ipynb` (NEW)

**Purpose**: Colab notebook for downloading and preprocessing ScanNet data.

**Cell structure** (modeled on omni_preprocess.ipynb):

| Cell | Title | Content |
|------|-------|---------|
| 0 | Title | Markdown: ScanNet Preprocessing Pipeline |
| 1 | Install & Imports header | Markdown |
| 2 | Install & Imports code | pip install, stdlib/third-party imports |
| 3 | Configuration header | Markdown |
| 4 | Configuration code | `BASE_OUT_DIR`, `TMP_DIR`, `DRIVE` paths, `FRAMES_PER_SCENE=66`, `MIN_FRAMES=10`, `TARGET_SIZE=256`, **`SCANNET_TOKEN`** (user must paste their download agreement token here), `MAX_WORKERS=4` (capped to avoid server rate-limiting and concurrent disk usage) |
| 5 | Scene Type Mapping header | Markdown |
| 6 | Scene Type Mapping code | `SCANNET_SCENE_TYPES` list (20 types, no Misc), `SCENE_TYPE_ALIASES` dict (initially empty — populated after Cell 14's metadata scan) |
| 7 | Mount Drive + Setup Tools header | Markdown |
| 8 | Mount Drive + Setup Tools code | Mount drive, download `download-scannet.py` and `SensorData.py` from ScanNet repo (place `SensorData.py` in a known dir on `sys.path`), download train/val split files (`scannetv2_train.txt`, `scannetv2_val.txt`) from GitHub |
| 9 | Download .txt Metadata (lightweight) header | Markdown |
| 10 | Download .txt Metadata code | Batch-download all `.txt` metadata files only (`--type .txt`). These are tiny (a few KB each) and safe to batch. Uses `SCANNET_TOKEN` for authentication. |
| 11 | Parse Scene Metadata header | Markdown |
| 12 | Parse Scene Metadata code | `parse_scene_metadata(txt_path)`: splits lines on first `=`, strips whitespace, returns dict. `normalize_scene_type(raw_str)`: lowercases, strips, checks aliases, returns canonical name or `None`. |
| 13 | Pre-Processing Metadata Scan header | Markdown |
| 14 | Pre-Processing Metadata Scan code | Parse ALL downloaded .txt files, print every unique raw `sceneType` string with counts. Identify which map to the 20 types, which need aliases, which will be skipped. **User reviews this output and updates `SCENE_TYPE_ALIASES` in Cell 6 before proceeding.** Also prints: total scenes in train+val lists, how many will be skipped (Misc/unknown), expected output frame count. |
| 15 | Frame Extraction Helpers header | Markdown |
| 16 | Frame Extraction Helpers code | `center_crop()`: pads with edge-replication if smaller than target (NOT zero-padding — avoids creating false depth=0 sentinels). `extract_frames_from_sens()`: loads `.sens` via `SensorData.py`, asserts `depth_shift == 1000.0` (sanity check), extracts raw uint16 depth arrays directly (no transformation — they're already millimeters), extracts RGB, center-crops both, saves as `.pt`. Raises `RuntimeError` on corrupt/unreadable `.sens` files. 120-second timeout per scene. |
| 17 | Download-Extract-Delete Micro-Loop header | Markdown: **CRITICAL: Colab disk constraint.** Raw .sens files are 100-200MB each. Each worker handles the full lifecycle for one scene: download → extract → delete. Pool capped at `MAX_WORKERS=4` to limit concurrent disk usage (~800MB peak) and avoid server rate-limiting. |
| 18 | Download-Extract-Delete Micro-Loop code | `process_scene(args)` where `args = (scene_id, split_name, txt_dir, output_base_dir, sens_tmp_dir, num_frames, target_size)`. Full lifecycle: (1) check `.done` marker → skip if present, (2) wipe partial files if no marker, (3) subprocess download with `SCANNET_TOKEN` and retry with exponential backoff on failure, (4) extract frames, (5) write `.done` marker, (6) `os.remove()` .sens file. `multiprocessing.Pool(MAX_WORKERS)` with `imap_unordered`. **Input scene list: ONLY scene IDs from `scannetv2_train.txt` + `scannetv2_val.txt`** — test scenes are never processed. `progress.json` is written ONLY by the main process (from `imap_unordered` results), never by workers. Progress bar tracks completed/skipped/failed. |
| 19 | Disk Check | `df -h` and `du -sh` |
| 20 | Organize into Split Directories header | Markdown |
| 21 | Organize into Split Directories code | Use official scannetv2_train.txt / scannetv2_val.txt to move processed tensors into train/<class>/ and val/<class>/ subdirectories |
| 22 | Validate Tensor Files header | Markdown |
| 23 | Validate Tensor Files code | Scan all .pt files, verify shapes (RGB: [3,256,256] uint8, Depth: [1,256,256] uint16), pairing, delete corrupt files |
| 24 | Data Leakage Test header | Markdown |
| 25 | Data Leakage Test code | Verify no scene appears in both train and val splits |
| 26 | Frame Diversity Verification header | Markdown |
| 27 | Frame Diversity Verification code | For a sample of scenes, compute pixel-difference between consecutive sampled frames, log per-scene frame counts and sampled indices. Note: pixel-difference is a heuristic sanity check, not a guarantee. |
| 28 | Dataset Statistics header | Markdown |
| 29 | Dataset Statistics code | Per-class distribution (train & val), frames per scene histogram, depth range stats, total sample counts |
| 30 | Normalization Stats header | Markdown |
| 31 | Normalization Stats code | Streaming (Welford) computation of RGB mean/std and depth mean/std from **train split only**. **Depth stats exclude pixels with value 0** (sentinel for missing data) to avoid biasing the mean downward. Depth values are converted to meters (`/ 1000.0`) before stats computation, matching the dataloader convention. |
| 32 | Write Metadata Files header | Markdown |
| 33 | Write Metadata Files code | Write class_names.txt, norm_stats.json, dataset_info.txt |
| 34 | Spot Check header | Markdown |
| 35 | Spot Check code | Visualize random samples from each class with matplotlib (RGB + depth side by side) |
| 36 | Package & Upload header | Markdown |
| 37 | Package & Upload code | tar.gz and copy to Drive |

### File 4: `tests/src/data_utils/test_scannet_pretrain_dataset.py` (NEW)

**Purpose**: Unit tests for the ScanNet dataset and dataloader factory.

### Implementation order

1. `src/data_utils/scannet_pretrain_dataset.py` -- the dataloader
2. `src/data_utils/__init__.py` -- register exports
3. `tests/src/data_utils/test_scannet_pretrain_dataset.py` -- tests
4. `notebooks/scannet_preprocess.ipynb` -- the preprocessing notebook

## 4. Interface Contracts

### `src/data_utils/scannet_pretrain_dataset.py`

#### `ScanNetPretrainDataset`

```python
class ScanNetPretrainDataset(Dataset):
    """ScanNet pretraining dataset.

    Same augmentation pipeline as OmniPretrainDataset except random erasing
    is spatially synchronized between RGB and depth (same box coordinates
    when both modalities are erased).

    Tensor format:
    - RGB: uint8 [3, 256, 256] on disk, returned as float32 [3, crop_size, crop_size]
    - Depth: uint16 [1, 256, 256] on disk (mm), returned as float32 [1, crop_size, crop_size] (meters)
    """

    VALID_SPLITS = ('train', 'val')

    def __init__(
        self,
        data_root: str,
        split: str,
        samples: list[tuple[str, str, int]],
        class_names: list[str],
        norm_stats: dict,
        crop_size: int = 224,
        normalize: bool = True,
        rgb_aug_prob: float = 1.0,
        rgb_aug_mag: float = 1.0,
        depth_aug_prob: float = 1.0,
        depth_aug_mag: float = 1.0,
    ):
        """Same signature as OmniPretrainDataset."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Same augmentation pipeline as OmniPretrainDataset.__getitem__
        except random erasing is synchronized:
        - RGB and depth independently decide WHETHER to erase
        - When both fire, erase params are sampled once and applied to both
        - When only one fires, it erases independently
        """

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights. Shape [num_classes], float32."""

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler. Shape [num_samples], float64."""

    def get_class_distribution(self) -> dict[str, dict[str, int | float]]:
        """Class distribution stats. {'class_name': {'count': int, 'percentage': float}}."""

    def get_norm_stats(self) -> dict:
        """Return normalization statistics dict."""
```

#### `get_scannet_pretrain_dataloaders`

```python
def get_scannet_pretrain_dataloaders(
    data_root: str = 'data/scannet_pretrain_256',
    batch_size: int = 32,
    num_workers: int = 4,
    crop_size: int = 224,
    use_class_weights: bool = False,
    seed: int = 42,
    normalize: bool = True,
    balanced_sampling: bool = True,
    rgb_aug_prob: float = 1.0,
    rgb_aug_mag: float = 1.0,
    depth_aug_prob: float = 1.0,
    depth_aug_mag: float = 1.0,
) -> tuple:
    """Same return signature as get_omnipretrain_dataloaders:
    (train_loader, val_loader, num_classes) if use_class_weights=False
    (train_loader, val_loader, num_classes, class_weights) if True

    Bug fix over Omni: val_workers = max(num_workers // 2, 0), and
    persistent_workers/prefetch_factor use val_workers (not num_workers).
    """
```

### Notebook key functions

```python
def parse_scene_metadata(txt_path: str) -> dict:
    """Parse <scene_id>.txt (key = value format, split on first '=').
    Returns dict of field->value. Key field: 'sceneType'."""

def normalize_scene_type(scene_type_str: str) -> str | None:
    """Normalize raw sceneType string to canonical class name.
    Lowercases, strips whitespace, checks SCENE_TYPE_ALIASES.
    Returns None if the type is Misc or unrecognized (scene should be skipped)."""

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop HxW or HxWxC array. If smaller than size, pads with
    edge replication (NOT zero-padding, to avoid false depth sentinels)."""

def extract_frames_from_sens(
    sens_path: str, output_dir: str, scene_id: str,
    num_frames: int = 66, target_size: int = 256,
) -> int:
    """Extract evenly-spaced frames from .sens file. Asserts depth_shift
    == 1000.0 (sanity check). Saves raw uint16 depth directly (already mm).
    Returns count extracted. Raises RuntimeError on corrupt .sens files."""

def process_scene(
    args: tuple[str, str, str, str, str, int, int],
) -> dict:
    """Full download-extract-delete micro-loop for one scene.
    args = (scene_id, split_name, txt_dir, output_base_dir,
            sens_tmp_dir, num_frames, target_size)

    (1) Check for <scene_id>.done marker — if present, return status='skipped'
    (2) If no marker but partial files exist, wipe and re-process
    (3) Download .sens via subprocess with retry + exponential backoff
    (4) Parse metadata, extract frames to .pt tensors (120s timeout)
    (5) Write <scene_id>.done marker after all tensors saved
    (6) os.remove() the .sens file immediately

    Returns dict with keys: scene_id, class_name, num_frames,
    status ('ok'/'skipped'/'error'), error_msg (if error)."""
```

## 5. Testing Strategy

### `_load_class_names(data_root)`
- Valid plain format -> returns correct list
- Indexed format ("0: bathroom") -> returns correct list
- Missing file -> raises `FileNotFoundError`
- Empty lines -> skipped

### `_load_norm_stats(data_root)`
- Valid JSON with all 4 keys -> returns dict
- Missing file -> raises `FileNotFoundError`

### `_discover_samples(data_root, class_names)`
- Two classes with 2 paired samples each -> returns 4 tuples sorted
- Unpaired RGB without depth -> raises `ValueError`
- Unpaired depth without RGB -> raises `ValueError`
- Folder not in class_names -> skipped
- Empty class folder -> returns 0 samples

### `ScanNetPretrainDataset.__init__`
- Invalid split "test" -> raises `ValueError`
- Valid construction -> sets all attributes

### `ScanNetPretrainDataset.__getitem__` (train)
- Returns correct shapes: rgb [3, 224, 224], depth [1, 224, 224], label int
- rgb and depth are float32
- Synchronized random erasing: when both RGB and depth are erased, detect the erased region (contiguous block of constant fill value) and assert the spatial mask is identical in both tensors

### `ScanNetPretrainDataset.__getitem__` (val)
- Returns CenterCropped output (deterministic)
- No augmentations applied
- Deterministic: two calls with same idx return identical tensors

### `get_scannet_pretrain_dataloaders`
- Valid data -> returns (train_loader, val_loader, num_classes)
- `use_class_weights=True` -> returns 4-tuple with class_weights as `torch.Tensor`
- Missing directory -> raises `FileNotFoundError`
- `num_workers=0` -> no crash (prefetch_factor=None)
- `num_workers=1` -> val_loader gets 0 workers without crash (bug fix verified)

### Integration test
- Fake data with 3 classes, 5 samples each
- Iterate one batch, verify shapes and label range

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ScanNet scene type strings inconsistent | Cell 14 prints all raw values before processing; SCENE_TYPE_ALIASES built from actual data; unrecognized types skipped |
| SensorData.py parsing failures | Per-scene try/except with 120s timeout, continue processing, report failures at end |
| Colab disk exhaustion | Strict download-extract-delete micro-loop; MAX_WORKERS=4 caps concurrent .sens files at ~800MB; disk check cell; budget: ~34GB tensors + ~25GB tar = ~60GB of ~200GB |
| Varying depth image dimensions | center_crop with edge-replication padding (not zero-padding, which would create false depth sentinels) |
| Near-duplicate sampled frames | Frame diversity verification cell; acknowledged as heuristic, primary guarantee is even-spacing |
| Scenes with too few frames | MIN_FRAMES=10 threshold; scenes below are skipped and logged |
| ScanNet download authentication | Config cell requires user to paste SCANNET_TOKEN; download fails clearly without it |
| Unexpected depth_shift value | extract_frames_from_sens asserts depth_shift == 1000.0; logs warning if different (raw uint16 values saved as-is since they're already mm with shift=1000) |
| Server rate-limiting | MAX_WORKERS=4 caps concurrent downloads; retry with exponential backoff on failure |
| progress.json corruption | Written only by main process from imap_unordered results; workers use .done marker files only |
| SensorData.py in multiprocessing | Placed in known sys.path dir; Colab uses fork (Linux default), so imports are inherited |

## 7. Review History

### Round 1 (2026-03-30)

**Reviewer findings addressed:**
- Clarified that helper functions are copied, not imported, from omnipretrain_dataset.py
- Specified synchronized erasing implementation: `v2.RandomErasing.get_params()` + `F2.erase()` with shared coordinates
- Fixed `get_class_distribution` return type: `dict[str, dict[str, int | float]]`
- Added val determinism test and `num_workers=1` safety test
- Capped multiprocessing pool to MAX_WORKERS=4
- Added ScanNet authentication requirement
- Fixed val_loader num_workers/prefetch_factor/persistent_workers bug

**Gap-finder findings addressed:**
- C1: Investigated depth_shift — it's a float divisor (1000.0), not a bitwise shift. Raw uint16 values are already mm. No transformation needed; added assertion as sanity check (Decision 6)
- C2: Capped concurrent downloads to MAX_WORKERS=4 with retry/backoff
- C3: progress.json written only by main process, workers use .done markers
- C4: Added SCANNET_TOKEN config cell for authentication
- I2: Micro-loop iterates only over train+val scene IDs
- I4: center_crop uses edge-replication padding, not zero-padding
- I5: Synchronized erasing implementation specified
- I6: SCENE_TYPE_ALIASES built from actual Cell 14 output
- I7: .txt parsing format specified (key = value, split on first =)
- M3: Norm stats exclude depth=0 sentinels
- M4: val_loader num_workers bug fixed
