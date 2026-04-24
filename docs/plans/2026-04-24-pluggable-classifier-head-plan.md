# Pluggable Classifier Head for LINet3 (Linear + Max-Out)

## Context

LINet3's current classifier is a single `nn.Linear(feature_dim, num_classes)`. Diagnostics on the SUN RGB-D 19-category benchmark show a 13.6pp val/test MCA gap (60.05% → 46.50%), with classes like `dining_area` dropping from 76% to 25% because val and test contain **different visual modes of the same semantic label** (bars/cafeterias vs. lab-adjacent/corridor/basement dining). Feature-space analysis (RBF MMD, Spearman correlation per-class) indicates features are roughly similar — it is the *decision boundary* that fails to transfer. A single linear hyperplane per class cannot accommodate multi-modal class structure that is unavoidable in fused multimodal representations.

**Proposed change:** factor the classifier into pluggable `nn.Module` heads (`LinearHead`, `MaxoutHead`), selected by a `classifier_head: str` kwarg on `LINet`. The max-out head learns K parallel hyperplanes per class and max-pools the resulting logits; gradient descent routes different visual modes to different sub-nodes automatically. Notebook switches to `classifier_head='maxout', num_subnodes=3`. Default stays `'linear'` so the baseline remains byte-identical.

Loss, optimizer, scheduler, samplers, mixup, SAM, label smoothing, and diagnostics all consume `(B, num_classes)` logits. Both heads return that shape; nothing downstream changes.

Intended outcomes:
- Recover accuracy on multi-modal classes (`dining_area`, `library`, `bedroom`).
- Preserve behavior on unimodal classes (one sub-node naturally dominates).
- Establish an extension point so future heads (cosine-prototype, ArcFace, MoE) plug in as a new `nn.Module` + one branch in `_build_network` — no further LINet API churn.
- Leave LINet's core multimodal-fusion contribution untouched.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Head structure | Each head is its own `nn.Module` (`LinearHead`, `MaxoutHead`) in a new file `src/models/linear_integration/li_net3/heads.py`. Each head encapsulates its own `nn.Linear` and any head-specific forward logic. LINet's `forward` calls `self.fc(features)` with no branching. | Adding a new head is a new class + one `elif` in `_build_network`, not edits to LINet's forward path. Each head is unit-testable in isolation. Head-specific parameters (`num_subnodes` for maxout, future `margin` for arcface) stay scoped to the head class rather than cluttering LINet's signature. |
| Head selection API | New kwargs on `LINet.__init__`: `classifier_head: str = 'linear'` and `num_subnodes: int = 3`. Stored on `self.classifier_head_type` (avoiding name collision with `self.fc` which is the head module instance) and `self.num_subnodes`. | `classifier_head` is the type discriminator; `num_subnodes` is the maxout-specific arg. `num_subnodes=3` default matches the notebook intent — it is silently ignored when `classifier_head='linear'` so the LINet default behavior is unchanged. |
| Validation | In `_build_network`: `if/elif/else` on `self.classifier_head_type` with an explicit `raise ValueError` on unknown head. No `num_subnodes` validation when head is `'linear'` (it's simply unused and stored for introspection). `MaxoutHead.__init__` asserts `num_subnodes >= 1`. | Matches the user-specified pattern: head-class-local validation keeps each head self-contained. |
| Factory exposure | Add explicit `classifier_head: str = 'linear'` and `num_subnodes: int = 3` to `li_resnet18` (used by notebook). Others (`li_resnet34/50/101/152`) already pass `**kwargs` so no change needed. | Explicit on the one factory in active use; implicit elsewhere to minimize diff. |
| Weight initialization | PyTorch default `nn.Linear` (Kaiming-uniform weights, zero bias). No symmetry-breaking noise needed. | Each sub-node starts with a different random hyperplane; argmax routing specializes them during training. Standard max-out init. |
| Reshape order in MaxoutHead | `logits.view(B, num_classes, num_subnodes)` where `fc.weight` is `(C·K, feat_dim)` laid out so rows `[c·K : (c+1)·K]` are class c's K sub-nodes. Use `.amax(dim=2)` (returns Tensor) not `.max(dim=2)` (returns namedtuple). | Consistent, contiguous, matches the spec; `.amax` keeps the forward graph clean. |
| Sub-node utilization monitoring | **Post-hoc diagnostic notebook cell** using a forward hook on `model.fc.fc` (the inner `nn.Linear` of `MaxoutHead`) that captures pre-max logits, reshapes via `model.fc.num_subnodes`, takes `argmax(dim=2)`, aggregates per-class. No changes to `fit()`. | `fit()` already has 40+ kwargs and this is a diagnostic, not a training signal. A one-shot pass at end of training delivers the same insight with zero risk to the training loop. |
| Bit-for-bit equivalence | Two complementary tests:<br>(a) **Head-level math equivalence**: build `LinearHead(feat, C)` and `MaxoutHead(feat, C, num_subnodes=1)`, copy fc weights, assert forward output `torch.equal`. Tests that the math of MaxoutHead reduces to LinearHead at K=1.<br>(b) **Model-level construction equivalence**: seed RNG, build `li_resnet18(num_classes=19)` and `li_resnet18(num_classes=19, classifier_head='maxout', num_subnodes=1)` from scratch, assert `torch.equal` on fc weights AND on a non-fc backbone weight AND on forward output. **No weight copying.** Tests that maxout-K=1 consumes the same RNG amount as the linear path so weights are naturally identical. | Test (a) is a pure unit test of the heads' math. Test (b) catches RNG-consumption drift in the construction path — copying weights would mask exactly the bug it's meant to catch. Both are needed because they verify different invariants. |
| Checkpoint compatibility | The notebook loads pretrained ScanNet weights with `strict=False` (cell 16 comment: "The fc head is skipped automatically since num_classes differs"). With the new head modules, the state_dict key for the head changes from `fc.weight` to `fc.fc.weight` (since `fc` is now `LinearHead`/`MaxoutHead` and the inner Linear is `fc.fc`). The pretrained checkpoint's `fc.weight` won't match any current key — it will be silently dropped. | This is *correct* (we never want to load the old single-linear classifier into a max-out head) but the silent-drop is a footgun. Mitigation: cell 21 explicit logging of `loaded_keys` vs `skipped_keys` (and any `unexpected_keys` from `load_state_dict(strict=False)` return value) so the first run after this change makes it obvious that `fc.weight`/`fc.bias` were dropped without breaking. |
| Naming | `num_subnodes` (not `K`) in code. Module name `MaxoutHead` (not `MaxOutHead`) following PyTorch single-cap convention (`MaxPool2d`, `Softmax`, `BatchNorm2d`). Attribute `self.classifier_head_type` (not `self.classifier_head`) to disambiguate from `self.fc` which is the head instance. | Discoverable, conventional, no collisions. |

### Why this is a clean, safe change
- **Output contract unchanged.** `model(streams)` still returns `(B, num_classes)` logits. Every downstream call site — `MultiStreamLoss`, `FocalLoss`, `mixup_loss`, SAM, label smoothing, MCA computation (`li_net.py:1937, 2083`), TTA, hook-parity check (cell 58), per-class diagnostics (cells 59–68) — is shape-compatible without edits.
- **Default preserves baseline.** `classifier_head='linear'` constructs `LinearHead(feat_dim, C)`, whose forward is `return self.fc(x)`. Identical to the current code path. `num_subnodes=3` default is unused when head is linear.
- **Forward stays simple.** `logits = self.fc(integrated_features)` — no `if` branches in the hot path. Head dispatch happens once at construction.
- **Parameter cost is trivial.** At `width_multiplier=0.75`, `feature_dim=384`. Linear head: 7,315 params. Maxout K=3: 21,945 params. Negligible vs. ~2.4M LINet18 backbone.

---

## Files to Modify

### 1. `src/models/linear_integration/li_net3/heads.py` (new file)

```python
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Standard linear classifier: nn.Linear(feat_dim, num_classes)."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MaxoutHead(nn.Module):
    """K-sub-node max-out classifier.

    Each class has K parallel sub-classifiers; the output logit for class c is
    the max over its K sub-node activations. Trained end-to-end with standard
    cross-entropy on the post-max logits. With K=1 this reduces mathematically
    to a standard linear classifier (max of a singleton is the singleton).
    """

    def __init__(self, feat_dim: int, num_classes: int, num_subnodes: int = 3):
        super().__init__()
        assert num_subnodes >= 1, f"num_subnodes must be >= 1, got {num_subnodes}"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_subnodes = num_subnodes
        self.fc = nn.Linear(feat_dim, num_classes * num_subnodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, C*K)
        return logits.view(-1, self.num_classes, self.num_subnodes).amax(dim=2)  # (B, C)
```

Both classes are tiny, dependency-free, and unit-testable in isolation. The package's existing modular structure (`conv.py`, `blocks.py`, `container.py`, `pooling.py`, `gradient_monitor.py`, `stream_monitor.py`) makes `heads.py` a natural fit.

### 2. `src/models/linear_integration/li_net3/__init__.py`

Add `LinearHead`, `MaxoutHead` to the package exports so they can be imported and tested directly.

### 3. `src/models/linear_integration/li_net3/li_net.py`

**Imports** (top of file)

Add: `from .heads import LinearHead, MaxoutHead`.

**`LINet.__init__`** (lines 78–120)

Add two kwargs after `width_multiplier` (line 93):
```python
classifier_head: str = 'linear',  # 'linear' | 'maxout' | future heads
num_subnodes: int = 3,            # only used when classifier_head == 'maxout'
```
Store on `self` before `super().__init__` (around line 101), alongside the other LINet-specific attributes:
```python
self.classifier_head_type = classifier_head
self.num_subnodes = num_subnodes
```
No validation here — the `if/elif/else` in `_build_network` handles unknown heads with `raise ValueError`, and `MaxoutHead.__init__` validates `num_subnodes`. When head is `'linear'`, `num_subnodes` is silently unused (it just sits on `self` for diagnostic introspection).

**`_build_network`** (lines 131–175)

Replace the final two lines (173–175):

```python
# Before
feature_dim = base[3] * block.expansion
self.fc = nn.Linear(feature_dim, self.num_classes)

# After
feature_dim = base[3] * block.expansion
if self.classifier_head_type == 'linear':
    self.fc = LinearHead(feature_dim, self.num_classes)
elif self.classifier_head_type == 'maxout':
    self.fc = MaxoutHead(feature_dim, self.num_classes, self.num_subnodes)
else:
    raise ValueError(
        f"Unknown classifier_head: {self.classifier_head_type!r}. "
        f"Must be 'linear' or 'maxout'."
    )
```

**`forward`** (lines 281–330)

**No change** to the call site:
```python
logits = self.fc(integrated_features)
return logits
```
Branching is gone; the head module handles its own forward logic. The current line 329 already does exactly this — no edit needed beyond what `_build_network` does to `self.fc`'s type.

Update the docstring `Returns:` line (298) to: `Classification logits [batch_size, num_classes] from integrated stream. The classifier head is selected by the classifier_head kwarg ('linear' or 'maxout'); both return (B, num_classes).`

**`li_resnet18` factory** (lines 2782–2819)

Add `classifier_head: str = 'linear'` and `num_subnodes: int = 3` to the signature, pass both through to `LINet(...)`. Update the docstring `Args:` block with one-line descriptions and a `classifier_head='maxout', num_subnodes=3` usage example.

Other factories (`li_resnet34/50/101/152`, lines 2821–2926) already accept `**kwargs` and forward to `LINet`, so the new kwargs flow through without changes. No edits required.

### 4. `notebooks/colab_LINet3_SUN_training_with_val.ipynb`

**Cell 16 — `MODEL_CONFIG`**

Add two lines to the `MODEL_CONFIG` dict:

```python
'classifier_head': 'maxout',  # 'linear' (baseline) | 'maxout'
'num_subnodes': 3,            # K: max-out sub-nodes per class (only used when classifier_head='maxout')
```

Inline comment above the dict explaining the knob: "classifier_head='linear' is the baseline single-hyperplane-per-class head; classifier_head='maxout' with num_subnodes=K gives K parallel hyperplanes per class with max-pooling, intended for multi-modal class structure. classifier_head='maxout', num_subnodes=1 reduces mathematically to 'linear'."

**Cell 21 — model construction**

Add both new kwargs to the `li_resnet18(...)` call (immediately after `use_amp`):

```python
classifier_head=MODEL_CONFIG['classifier_head'],
num_subnodes=MODEL_CONFIG['num_subnodes'],
```

Update the print block at the bottom of cell 21 to include:
- `Classifier head: {MODEL_CONFIG['classifier_head']}`
- `Sub-nodes per class: {MODEL_CONFIG['num_subnodes']}` (only meaningful when head is `'maxout'`)
- `Classifier head params: {sum(p.numel() for p in model.fc.parameters()):,}` — uses `model.fc.parameters()` so it works whether `model.fc` is `LinearHead` or `MaxoutHead`.

**Pretrained-load logging + hard abort — modify the existing checkpoint-load block in cell 21** (the `LOAD_PRETRAINED` branch) to capture the explicit return value from `load_state_dict(..., strict=False)` and **hard-abort** on any mismatch beyond the expected fc-key swap:

```python
result = model.load_state_dict(pretrained_sd, strict=False)
print("Pretrained load:")
print(f"  missing    ({len(result.missing_keys)}): {result.missing_keys[:8]}{'...' if len(result.missing_keys) > 8 else ''}")
print(f"  unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:8]}{'...' if len(result.unexpected_keys) > 8 else ''}")

# Hard abort on any mismatch beyond the expected classifier-head key swap.
# After the head refactor, the inner Linear sits at `fc.fc.{weight,bias}`,
# while the pretrained checkpoint has the old `fc.{weight,bias}`. This is the
# only legal mismatch — anything else means a backbone key drifted.
expected_missing    = {'fc.fc.weight', 'fc.fc.bias'}
expected_unexpected = {'fc.weight', 'fc.bias'}
missing_surprises    = set(result.missing_keys)    - expected_missing
unexpected_surprises = set(result.unexpected_keys) - expected_unexpected
if missing_surprises or unexpected_surprises:
    raise RuntimeError(
        "Pretrained load: unexpected state_dict mismatch beyond classifier head.\n"
        f"  unexpected backbone keys in checkpoint: {sorted(unexpected_surprises)}\n"
        f"  missing backbone keys in model:         {sorted(missing_surprises)}\n"
        "Backbone weights have not loaded correctly — abort and investigate."
    )
```

This makes the "invalid run — abort" case an actual abort rather than a printed message that could be skimmed past. Expected behavior: `missing == {'fc.fc.weight', 'fc.fc.bias'}` and `unexpected == {'fc.weight', 'fc.bias'}` (the head-refactor footprint), nothing else.

**New cell (insert after cell 68, before the data-pipeline-parity section at cell 69)** — sub-node utilization diagnostic

A single diagnostic cell that:
1. Asserts `model.classifier_head_type == 'maxout'` (skip the cell with a printed message if linear, since utilization is meaningless for K=1).
2. Re-uses the `hook_capture`-style pattern from cell 58, but registers a forward hook on **`model.fc.fc`** (the inner `nn.Linear` of `MaxoutHead`) to capture pre-max logits.
3. Reshapes captured logits via `pre_max = fc_output.view(B, model.fc.num_classes, model.fc.num_subnodes)`.
4. Runs over the existing `hpo_train` eval loader (already in `eval_loaders`, shuffle/augmentation OFF — same loader cells 57/58 use, so the parity check applies).
5. For each sample, computes `subnode_idx = torch.argmax(pre_max, dim=2)` — the same index the in-model `.amax(dim=2)` would have selected. Aggregate by GT class for utilization-by-class.
6. Builds a `(num_classes, num_subnodes)` utilization count matrix.
7. Reports:
   - Per-class sub-node utilization distribution (counts and fractions).
   - **Per-class utilization entropy** (`H_c = -Σ p_ck log p_ck`, normalized by `log K`) plotted as a bar chart with y-axis label "Sub-node usage entropy per class (normalized to [0, 1])". H≈1 → fully utilized (truly multi-modal), H≈0 → one sub-node dominates (effectively K=1 for that class).
   - Dead sub-node table: classes with any sub-node used <5% of the time, called out by name.
   - Repeat the same analysis on the `hpo_val` and `official_test` loaders for cross-split comparison — if a class shows high entropy on train but routes everything to one sub-node on test, that is itself diagnostic of the same multi-mode-mismatch issue the change is meant to address.
8. Saves results to `results/maxout_subnode_utilization_{timestamp}.json` with this schema:
   ```json
   {
     "config": {"classifier_head": "maxout", "num_subnodes": 3,
                "num_classes": 19, "feature_dim": 384,
                "seed": ..., "sampler_variant": "v3", "use_mixup": false,
                "use_sam": false, "label_smoothing": 0.06, "epochs_run": 17,
                "checkpoint_path": "..."},
     "per_split": {
       "hpo_train": {"counts": [[c×K matrix]], "entropy_per_class": [...],
                     "dead_subnodes": {"class_name": [subnode_idx, ...]}},
       "hpo_val":   {...},
       "official_test": {...}
     }
   }
   ```
   The `config` block is critical — without it the JSON floats free of the run that produced it.

**Note on dead-sub-node trajectory.** A sub-node can become "partially dead" (wins early then never again) and decay further via weight decay. End-of-training utilization is a reasonable proxy. If you have per-epoch checkpoints (the existing `checkpoint_epochs` mechanism in `fit()` supports this), the same diagnostic cell can iterate over them to plot the utilization trajectory. Don't add this preemptively; only if end-of-training results show concerning dead counts. Standard fix if needed: a small inter-sub-node diversity penalty as a new optional loss term — but only add if diagnostics show it's required.

No changes to `fit()`, no changes to the model, and no training-time overhead.

### 5. `tests/test_classifier_head.py` (new file)

Mirror the style of `tests/test_auxiliary_classifiers.py` and `tests/src/models/core/test_resnet.py`. Eight tests, **grouped in the file in this exact order** (head-module → LINet-integration → behavioral) so failure context is clear and related tests are co-located:

**Group A — Head-module tests** (operate on `LinearHead` and `MaxoutHead` directly, no LINet backbone):

1. **`test_linear_head_shape`** — `LinearHead(384, 19)`; forward `(B=4, 384)`; assert output shape `(4, 19)` and `head.fc.weight.shape == (19, 384)`.

2. **`test_maxout_head_shape`** — `MaxoutHead(384, 19, num_subnodes=3)`; forward `(B=4, 384)`; assert output shape `(4, 19)` (NOT `(4, 19, 3)`) and `head.fc.weight.shape == (57, 384)`. Also assert `head.num_subnodes == 3` and `head.num_classes == 19` (introspection attributes used by the diagnostic cell).

3. **`test_maxout_head_rejects_zero_subnodes`** — `MaxoutHead(384, 19, num_subnodes=0)` raises `AssertionError`.

4. **`test_maxout_k1_math_equals_linear`** — head-level math equivalence with explicitly matched weights:
   ```python
   linear = LinearHead(384, 19)
   maxout_k1 = MaxoutHead(384, 19, num_subnodes=1)
   maxout_k1.fc.weight.data.copy_(linear.fc.weight.data)
   maxout_k1.fc.bias.data.copy_(linear.fc.bias.data)
   x = torch.randn(8, 384)
   assert torch.equal(linear(x), maxout_k1(x))
   ```
   Tests that `MaxoutHead`'s `view + amax` over a singleton K dimension is mathematically identity. Copying weights here is correct — this is a forward-math test, not an equivalence-of-construction test.

**Group B — LINet integration tests:**

5. **`test_default_is_linear_head`** — `li_resnet18(num_classes=19)`; assert `model.classifier_head_type == 'linear'` (the attribute name set by LINet, not `model.classifier_head` which would collide with the module instance), `isinstance(model.fc, LinearHead)`, and `model.fc.fc.weight.shape == (19, feature_dim)`. Forward output shape `(B, 19)`.

6. **`test_unknown_head_raises`** — `li_resnet18(num_classes=19, classifier_head='unknown')` raises `ValueError` whose message contains both the unknown head string and the valid options (`'linear'`, `'maxout'`).

7. **`test_maxout_k1_strict_construction_equals_linear`** — strict construction equality, **no weight copying**:
   ```python
   torch.manual_seed(0); torch.cuda.manual_seed_all(0)
   m_linear = li_resnet18(num_classes=19)  # defaults: linear head
   torch.manual_seed(0); torch.cuda.manual_seed_all(0)
   m_maxout_k1 = li_resnet18(num_classes=19, classifier_head='maxout', num_subnodes=1)
   # Both heads' inner Linear should have identical weights via RNG consumption order
   assert torch.equal(m_linear.fc.fc.weight, m_maxout_k1.fc.fc.weight)
   assert torch.equal(m_linear.fc.fc.bias, m_maxout_k1.fc.fc.bias)
   # Backbone weight equality confirms no upstream RNG drift
   assert torch.equal(m_linear.conv1.stream_weights[0], m_maxout_k1.conv1.stream_weights[0])
   # Forward equality on identical input
   torch.manual_seed(123)
   streams = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
   m_linear.eval(); m_maxout_k1.eval()
   assert torch.equal(m_linear(streams), m_maxout_k1(streams))
   ```
   This proves the maxout-K=1 construction path consumes the same RNG as the linear path (i.e., the only RNG consumer in either head is `nn.Linear(feat, 19)` either way, since `19 * 1 == 19`). If a future change adds RNG-consuming logic to `MaxoutHead.__init__` (e.g., per-sub-node init), this test fails immediately.

**Group C — Behavioral test (last in file):**

8. **`test_maxout_k3_argmax_specialization`** (`@pytest.mark.slow`) — synthetic 2-modal-per-class dataset where K=1 *cannot* achieve low loss but K=2+ can:
   - **Setup**: train heads in isolation (no LINet backbone). Construct 384-dim features by sampling from XOR-like Gaussian clusters: class A's two clusters at `(+x, +y, …)` and `(-x, -y, …)` in two leading dims (rest noise); class B's two at `(+x, -y, …)` and `(-x, +y, …)`. A single linear hyperplane per class gets ~50%; max-out with K=2+ can solve it.
   - **Precondition**: train `LinearHead` and `MaxoutHead(num_subnodes=3)` on the same data for 200+ steps with the same optimizer; assert `loss_maxout < 0.5 * loss_linear`. If this fails, the test setup is wrong (clusters too close, feature dim too large, training too short) — fix the setup, don't relax the assertion.
   - **Behavioral assertion**: with the trained `MaxoutHead`, run inference on cluster-1 samples and cluster-2 samples of the same class; capture pre-max logits via the `.fc(x).view(B, C, K)` path; assert `argmax(dim=2)` for the GT-class column differs between clusters >70% of the time. This is the test that K=3 actually specialized sub-nodes to modes, not just learned a redundant copy.

Run with `pytest tests/test_classifier_head.py -v` (excludes slow) or `pytest tests/test_classifier_head.py -v -m slow` (slow only).

### 6. Pre-implementation `model.fc` audit (already done — diff surface is small)

A repo-wide grep for `\.fc\.weight`, `\.fc\.bias`, `\.fc\.parameters`, `\.fc\.register`, `model\.fc`, and `self\.fc` was run before finalizing this plan. Categorized results:

**Sites that work unchanged** (head is callable, `.parameters()` works on any `nn.Module`, hooks fire on the head module and capture *input* = penultimate features):
- `src/models/linear_integration/li_net3/li_net.py:329` — forward call `logits = self.fc(integrated_features)`. Stays as-is.
- `src/models/linear_integration/li_net3/stream_monitor.py` — 4 sites of `self.model.fc(features)`. Head is callable; works.
- `notebooks/colab_LINet3_SUN_training_with_val.ipynb:1573, 1351 (commented), 4664, 5607` — `model.fc.parameters()` (works on any nn.Module), `hook_capture(model, loader, hook_module=model.fc)` (hook captures *input* to head = penultimate features, head-type-independent), `model.fc.register_forward_hook(_h)` (same — captures input).

**Sites that MUST be updated** (direct `.weight`/`.bias` access on `model.fc`, only valid when `model.fc` is `nn.Linear`):
- `tests/diagnose_gradient_diff.py:212` — `model_linet3.fc.weight.grad` → `model_linet3.fc.fc.weight.grad`.
- `tests/test_conv1_gradient_flow.py:238` — `fc_grad = model.fc.weight.grad` → `model.fc.fc.weight.grad`.

**Sites NOT affected by this change** (different model variant — not `li_net3`):
- `src/models/linear_integration/li_net.py` (the older non-`li_net3` LINet — separate file)
- `src/models/direct_mixing_*/dm_net.py`, `src/models/multi_channel/mc_resnet.py`, `src/models/core/resnet.py`, `src/models/linear_integration/li_net3_soma/`, `src/models/utils/stream_monitor.py`, `src/utils/visualization/stream_visualization.py`
- `tests/test_stream_monitoring_safety.py` (imports `from src.models.linear_integration.li_net`, NOT `li_net3`)
- `tests/src/models/core/test_resnet.py` (tests `core/resnet.py`)
- All NYU notebooks and `colab_LiNet3_SUN_hype_tune*.ipynb` baselines that mutate `model.fc` directly — these use other model variants. Not in scope.

**Total LINet3 update surface beyond the planned changes: 2 lines in 2 test files.** Both are line-for-line `.fc.weight` → `.fc.fc.weight` substitutions and should be done in the same commit as the head refactor.

**Other no-touch verifications** (downstream of `(B, C)` logits, refactor-orthogonal):
- `src/models/abstracts/abstract_model.py` — `compile()`, `fit()` operate on `(B, C)` logits.
- `src/training/losses.py`, `mixup.py`, `sam.py`, `samplers.py`, `optimizers.py`, `schedulers.py`, `modality_dropout.py` — all shape-compatible.
- `li_net.py:1937, 2083` MCA loops — iterate over `self.num_classes`; no head awareness needed.
- Cells 55–68 (per-class diagnostics, hook-parity, MMD, feature drift, bootstrap CI) — operate on `(B, C)` logits or penultimate features; both unchanged.

**Quick `num_classes` sanity grep** (still recommended before writing code):
```bash
grep -rn "num_classes" src/training/ src/models/abstracts/abstract_model.py
```
Confirm: no hardcoded `19`; `nn.CrossEntropyLoss(label_smoothing=...)` infers class count from logits shape; `mixup_loss`/`mixup_batch` operate on `(B,)` labels and `(B, C)` logits with no class-count assumption; no `K` variable collision with `num_subnodes`. Expected outcome: nothing to change.

---

## Critical Code Paths (for reviewers)

| What | Path | Line |
|---|---|---|
| Head module classes | `src/models/linear_integration/li_net3/heads.py` | new file |
| Package exports | `src/models/linear_integration/li_net3/__init__.py` | add `LinearHead`, `MaxoutHead` |
| Classifier construction | `src/models/linear_integration/li_net3/li_net.py` | 173–175 (replaced with if/elif/else dispatch) |
| Forward classifier call (unchanged) | `src/models/linear_integration/li_net3/li_net.py` | 329–330 |
| `LINet.__init__` signature | `src/models/linear_integration/li_net3/li_net.py` | 78–120 |
| `_build_network` called before `_initialize_weights` | `src/models/abstracts/abstract_model.py` | 86–92 |
| `li_resnet18` factory | `src/models/linear_integration/li_net3/li_net.py` | 2782–2819 |
| Notebook MODEL_CONFIG | `notebooks/colab_LINet3_SUN_training_with_val.ipynb` | cell 16 |
| Notebook model construction + load logging | `notebooks/colab_LINet3_SUN_training_with_val.ipynb` | cell 21 |
| Hook-parity reference (for diagnostic cell) | `notebooks/colab_LINet3_SUN_training_with_val.ipynb` | cell 58 |

---

## Verification

### Verification gate ordering (run in this order — each gate must pass before the next)

**Gate 1: Strict unit-test equality.**
```bash
pytest tests/test_classifier_head.py -v -m "not slow"
```
All seven non-slow tests must pass. Of particular note: `test_maxout_k1_strict_construction_equals_linear` uses `torch.equal` (not `allclose`) on fc weights, biases, and forward output. Hard fail if any byte differs — this catches RNG-consumption drift before any training runs.

**Gate 2: Existing test suite green.**
```bash
pytest tests/test_linet3_end_to_end.py -v
pytest tests/ -v -m "not slow"
```
No regression in any existing test under default `classifier_head='linear'`. If anything broke, it's almost certainly a `model.fc.weight`-style direct access that should be `model.fc.fc.weight` post-refactor — fix the call site.

**Gate 3: Per-component smoke tests for K=3.** Run *individually* before any full training, not all together:
- 3a. Plain K=3 + cross-entropy (no mixup, no SAM): 1 epoch, 100 steps → loss decreases, no NaN.
- 3b. K=3 + mixup enabled: same.
- 3c. K=3 + SAM enabled: same.
- 3d. K=3 + mixup + SAM: same.

Each smoke run only needs to confirm the loss curve is sane (monotone-ish decrease, no NaN/Inf, train_acc > 1/19 chance level by step 100). Any failure here points to an interaction issue caught early, not deep into a real ablation.

**Gate 4: K=1 end-to-end matches baseline within ±0.1pp.** Full training run with `classifier_head='maxout', num_subnodes=1`. Final `val_mca` must match the last committed baseline within `±0.1pp` across seeds (stricter than ±0.3pp — looser gates hide subtle bugs). If outside ±0.1pp, the maxout-K=1 path differs from baseline in a way that affects optimization — investigate before proceeding. (Equivalent run with `classifier_head='linear'` should be exactly bit-for-bit since LinearHead's forward is `return self.fc(x)`.)

**Gate 5: K=3 full run.** Record: `train_mca`, `val_mca`, per-class accuracy on `hpo_val` and `official_test`, confusion matrix, MMD per class, sub-node utilization JSON.
- **Primary success signal (hard):** test MCA improvement over baseline, and `dining_area` test accuracy > baseline's 25%. These are the metrics that directly express the intervention's intent.
- **Diagnostic signal (soft, not a hard gate):** val/test confusion-overlap top-3 — moving from 0/3 toward higher values would be encouraging, but confusion overlap is noisy at the top-3 level (a single confusion-class shift moves the metric). If primary signals are positive but this one isn't, the intervention is still likely working.
- **Secondary signal:** sub-node utilization cell shows at least some classes (especially `dining_area`, `library`, `bedroom`) using >1 sub-node with >10% frequency, with normalized entropy >0.3. If ALL classes are effectively K=1 (one dominant sub-node, entropy ~0), the head is not specializing — flag for investigation (likely insufficient specialization time, or the features are already linearly separable for most classes).
- Hook-parity check (cell 58) must still pass (<0.1pp divergence).

### Mini K-sweep (if K=3 shows improvement at Gate 5)
Sweep `num_subnodes ∈ {1, 2, 3, 4}` with fixed seed, same epochs. Report val MCA, test MCA, val–test gap, dead-sub-node count, mean normalized entropy. Expected shape: monotone improvement 1→2→3, flattening or slight regression at 4 (symptom: higher dead-sub-node count).

### Smoke check during implementation
After editing `li_net.py` and creating `heads.py`, in a Python REPL:
```python
from src.models.linear_integration.li_net3 import li_resnet18, LinearHead, MaxoutHead
import torch
m = li_resnet18(num_classes=19, classifier_head='maxout', num_subnodes=3,
                width_multiplier=0.75, dropout_p=0.0, device='cpu')
streams = [torch.randn(2, 3, 224, 224), torch.randn(2, 1, 224, 224)]
out = m(streams)
assert out.shape == (2, 19), out.shape
assert isinstance(m.fc, MaxoutHead), type(m.fc)
assert m.fc.fc.weight.shape == (57, 384)
print("OK:", out.shape, m.fc.fc.weight.shape)
```

---

## Out of Scope (per spec)

- Alternative heads (cosine-prototype, ArcFace, MoE) — but the head-module pattern makes them trivial future additions.
- Classifier-head ensembling.
- Learned K per class.
- Changes to feature extractor (width, depth, integration mechanism).

---

## Summary of Diff Surface

| File | Lines touched | Nature |
|---|---|---|
| `src/models/linear_integration/li_net3/heads.py` | New file, ~30 lines | `LinearHead` + `MaxoutHead` modules. |
| `src/models/linear_integration/li_net3/__init__.py` | +2 exports | Package surface for tests + future heads. |
| `src/models/linear_integration/li_net3/li_net.py` | ~12 lines added/modified (import, init kwargs + attribute store, `_build_network` dispatch, factory kwargs + doc, docstring) | Forward pass *unchanged*. Head dispatch happens once at construction. |
| `notebooks/colab_LINet3_SUN_training_with_val.ipynb` | Cell 16 (+2 config lines), cell 21 (+2 kwargs + log lines + load-key logging), +1 new diagnostic cell | Additive. |
| `tests/test_classifier_head.py` | New file, ~250 lines | 8 tests, one marked slow. Grouped: head-module → LINet integration → behavioral. |
| `tests/diagnose_gradient_diff.py` | 1 line | `model_linet3.fc.weight.grad` → `model_linet3.fc.fc.weight.grad`. |
| `tests/test_conv1_gradient_flow.py` | 1 line | `model.fc.weight.grad` → `model.fc.fc.weight.grad`. |

No deletions. No API renames. Default behavior is byte-identical to the current baseline (LinearHead's forward is `return self.fc(x)` over the same `nn.Linear`).
