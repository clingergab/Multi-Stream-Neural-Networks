# ADR-002: Pluggable Classifier Head for LINet3 (Linear + Max-Out)

**Date**: 2026-04-24
**Status**: Accepted
**Plan**: docs/plans/2026-04-24-pluggable-classifier-head-plan.md

## Context

LINet3's classifier was a bare `nn.Linear(feature_dim, num_classes)` assigned directly to
`self.fc`. Per-class diagnostics on the SUN RGB-D 19-category benchmark (cells 55–68 of
`notebooks/colab_LINet3_SUN_training_with_val.ipynb`) revealed a 13.6pp val/test MCA gap
(60.05% → 46.50%). Classes like `dining_area` fell from 76% val accuracy to 25% test
accuracy because val and test samples are drawn from structurally different visual modes of
the same semantic label — bars and cafeterias vs. lab-adjacent and corridor dining areas.

Feature-space analysis (RBF MMD, Spearman correlation per class) showed that features were
roughly similar across splits. The problem was the decision boundary: a single linear
hyperplane per class cannot accommodate multi-modal class structure that is unavoidable when
fusing RGB, depth, and HHA streams into a shared representation.

LINet is designed as a general multimodal fusion architecture, not a SUN RGB-D-specific one.
Hardwiring a single linear classifier as the only option is a restrictive inductive bias —
"each class has exactly one representative direction in feature space" — that mismatches the
architecture's purpose. The change that addressed the `dining_area` failure also needed to
establish an extension point for future heads (cosine-prototype, ArcFace, mixture-of-experts)
without requiring further LINet API churn.

## Decision

The classifier is factored into pluggable `nn.Module` heads in a new file
`src/models/linear_integration/li_net3/heads.py`. Two heads are provided: `LinearHead`
(a thin wrapper around `nn.Linear`) and `MaxoutHead` (`nn.Linear(feat, C·K)` followed by
`view(B, C, K)` and `.amax(dim=2)`). LINet selects the head via two new kwargs —
`classifier_head: str = 'linear'` and `num_subnodes: int = 3` — dispatched once in
`_build_network` via `if/elif/else` with `raise ValueError` on an unknown head. LINet's
forward is unchanged: `logits = self.fc(integrated_features)` — no hot-path branching.
The default is `'linear'`, preserving byte-identical behavior relative to the pre-refactor
baseline.

## Alternatives Considered

### Alternative A: Inline branching in LINet.forward
- **Pros**: No new classes or files; all logic stays in `li_net.py`.
- **Cons**: Adds a Python conditional to the forward hot path. Head-specific logic is split across `_build_network` (construction) and `forward` (inference). Adding a third head requires editing both methods.
- **Why rejected**: Maintainability degrades with each additional head. The `nn.Module` pattern keeps construction and forward logic co-located in the head class; LINet's forward stays a single unconditional call.

### Alternative B: MaxoutHead with num_subnodes=1 as the unified baseline
- **Pros**: One class instead of two; `LinearHead` is not strictly necessary.
- **Cons**: The intent of the single-hyperplane baseline becomes implicit — a reader seeing `MaxoutHead(..., num_subnodes=1)` must know that K=1 is the linear special case. Removes the ability to express "this model uses a plain linear classifier" without looking at the `num_subnodes` value.
- **Why rejected**: Making the type discriminator explicit (`classifier_head='linear'`) is clearer for both code readers and introspection. `LinearHead` also future-proofs the API: if `MaxoutHead` gains K-specific initialization logic, the K=1 path would silently acquire that behavior too.

### Alternative C: Auxiliary mixture-of-experts head with learned per-class gating
- **Pros**: More expressive than max-out; gating can route based on input features rather than simple argmax.
- **Cons**: Requires a gating network, auxiliary loss terms, and extra hyperparameters. Implementation surface is an order of magnitude larger than max-out.
- **Why rejected**: Max-out achieves the same "multiple sub-classifiers per class" effect with two lines of code and no additional loss terms. The hypothesis being tested is whether multi-modal decision boundaries help at all — the simplest intervention that tests that hypothesis should be tried first.

### Alternative D: Per-class learned K (variable sub-nodes per class)
- **Pros**: Allocates capacity where it is needed; unimodal classes use K=1 automatically.
- **Cons**: Non-trivial to implement; K must be treated as a discrete hyperparameter per class, requiring either Gumbel-softmax tricks or an outer HPO loop over a 19-dimensional discrete space.
- **Why rejected**: Fixed global K is standard in scene-classification literature and sufficient for the hypothesis test. Per-class K can be revisited if end-to-end results show that most classes are wasting sub-nodes.

## Consequences

### Positive
- Adding a new head is a new `nn.Module` class and one `elif` branch in `_build_network`.
  LINet's forward path and all downstream code — `MultiStreamLoss`, `FocalLoss`, `mixup_loss`,
  SAM, label smoothing, MCA loops, per-class diagnostics — require no changes because both
  heads return `(B, num_classes)` logits.
- The `LinearHead` default is byte-identical to the pre-refactor baseline.
  `li_resnet18(num_classes=19)` and `li_resnet18(num_classes=19, classifier_head='maxout',
  num_subnodes=1)` constructed from the same seed produce `torch.equal` fc weights, backbone
  weights, and forward output — no weight copying — because both paths consume the same RNG
  amount (`nn.Linear(feat, 19)` and `nn.Linear(feat, 19*1)` are the same call).
- Each head is unit-testable in isolation. `tests/test_classifier_head.py` includes four
  head-module tests that run without a LINet backbone.
- Parameter overhead of K=3 max-out at `width_multiplier=0.75` (feature dim 384) is
  21,945 vs. 7,315 for linear — negligible against the ~2.4M-parameter LINet18 backbone.

### Negative
- The state-dict key for the classifier changes from `fc.weight`/`fc.bias` to
  `fc.fc.weight`/`fc.fc.bias` because `self.fc` is now a `LinearHead` or `MaxoutHead` module
  whose inner `nn.Linear` is `self.fc.fc`. Pre-refactor checkpoints loaded with `strict=False`
  will silently drop their `fc.*` keys without error. Mitigation: the notebook's pretrained-load
  block (cell 21) now captures the `load_state_dict` return value, prints `missing_keys` and
  `unexpected_keys`, and hard-aborts via `RuntimeError` if anything beyond the expected
  `{fc.weight, fc.bias}` → `{fc.fc.weight, fc.fc.bias}` swap appears in the mismatch sets.
- Any code that previously accessed `model.fc.weight` directly must be updated to
  `model.fc.fc.weight`. The audit identified two such sites:
  `tests/diagnose_gradient_diff.py:212` and `tests/test_conv1_gradient_flow.py:238`. Both
  were updated in the same commit. A third site, `tests/src/models/abstracts/test_param_groups.py`,
  was also updated. All other call sites use the head as a callable or iterate `.parameters()`
  generically and required no changes.

### Neutral
- `num_subnodes` is stored on the LINet instance even when the head is `'linear'`, so
  diagnostic and introspection code can always read it. It is unused in the linear branch.
- The `self.classifier_head_type` attribute name (not `self.classifier_head`) was chosen to
  avoid collision with `self.fc`, which is the constructed head module instance.
- Gates 4 (K=1 end-to-end must match baseline within ±0.1pp) and 5 (K=3 full training run
  on SUN RGB-D, primary signal: test `dining_area` accuracy > baseline's 25%) require a GPU
  machine with the SUN RGB-D dataset and have not been executed at the time this ADR was
  written. This ADR documents the API design; experimental outcomes should be documented
  separately once available.

## Related
- docs/plans/2026-04-24-pluggable-classifier-head-plan.md — full implementation plan
- docs/decisions/2026-03-20-001-omnipretrain-dataset.md — prior ADR; established the project's
  pluggable-module convention
- src/models/linear_integration/li_net3/heads.py — `LinearHead` and `MaxoutHead` modules
- src/models/linear_integration/li_net3/li_net.py — head dispatch in `_build_network`
- tests/test_classifier_head.py — 8 tests covering head math, construction equivalence,
  integration, and K=3 sub-node specialization on synthetic multi-modal data
