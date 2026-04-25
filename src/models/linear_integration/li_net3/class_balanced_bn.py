"""Class-balanced BatchNorm for LINet3 integrated pathway.

Motivation
----------
Cell 19.16 (AdaBN diagnostic) on the mixup+SAM checkpoint showed that
swapping BN running statistics for batch statistics at test time recovers
+4.95pp test MCA on the 7 REPRESENTATION-drift classes (dining_area,
discussion_area, home_office, lab, lecture_theatre, library, study_space).
The cost was a -1.31pp regression on top-line accuracy because batch stats
hurt majority classes whose running stats were already well-fit.

Class-balanced BN attacks the same mechanism at training time: instead of
letting the running stats accumulate batch means/vars that are biased toward
majority classes, we accumulate UNWEIGHTED averages of per-class batch means/
vars. Minority classes contribute equally to the running statistics, so at
test time the running stats represent the per-class balanced feature
distribution rather than the dataset-frequency-weighted one.

Scope
-----
This implementation class-balances ONLY the integrated pathway of
``LIBatchNorm2d``. Per-stream pathways (RGB, depth) keep standard BN
behavior. Two reasons:

1. The integrated stream feeds the classifier — that is where the rep-drift
   mismatch most directly causes test errors.
2. Per-stream pathways have partial-blanking (modality dropout) which slices
   the batch via ``active_idx``; correctly slicing labels in lockstep adds
   surface area without clear evidence that per-stream CBBN matters here.

If integrated-only CBBN delivers a meaningful gain, per-stream CBBN can be
added later.

How labels reach the BN
-----------------------
Class labels do not flow through the standard model forward signature, so
they are passed via a thread-local stash set by ``stash_labels(...)`` in the
training loop. Each forward inside the context sees the labels; outside the
context (or in eval mode), the module falls through to standard BN behavior.

Math
----
Training-mode forward, labels stashed:

    For each class c present in batch:
        x_c = x[labels == c]                   # subset over class c
        μ_c = mean(x_c, dim=(N_c, H, W))       # (C,)
        σ²_c = var(x_c, dim=(N_c, H, W), unbiased=False)

    μ_bal = mean over c of μ_c                  # equal class weighting
    σ²_bal = mean over c of σ²_c
    y = γ ((x - μ_bal) / √(σ²_bal + ε)) + β

    running_mean ← (1−m)·running_mean + m·μ_bal
    running_var  ← (1−m)·running_var  + m·σ²_bal · n_total/(n_total−1)

Classes with fewer than 2 elements (samples × spatial) in the batch are
skipped from the per-class average. If no class qualifies, the pathway falls
back to standard batch stats for that batch.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conv import LIBatchNorm2d


# Thread-local stash for the current batch's labels.
_STASH = threading.local()


@contextlib.contextmanager
def stash_labels(labels: Optional[Tensor]):
    """Make ``labels`` visible to ClassBalancedLIBatchNorm2d during the wrapped
    forward. Outside the context (or with labels=None) modules behave like
    standard LIBatchNorm2d. Always restores the previous stash on exit.
    """
    prev = getattr(_STASH, "labels", None)
    _STASH.labels = labels
    try:
        yield
    finally:
        _STASH.labels = prev


def _peek_labels() -> Optional[Tensor]:
    return getattr(_STASH, "labels", None)


def _class_balanced_batch_stats(
    x: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, int]:
    """Compute per-class mean/var over (N_c, H, W), then average with equal
    weight across classes that have ≥2 elements (samples × H × W).

    Returns (balanced_mean, balanced_var, n_total). Falls back to standard
    batch stats with a warning if no class qualifies (extremely rare —
    requires <2 spatial elements * 0 samples in every present class).
    """
    if x.dim() != 4:
        raise ValueError(f"expected 4D input (N, C, H, W); got {x.shape}")
    if labels.dim() != 1 or labels.shape[0] != x.shape[0]:
        raise ValueError(
            f"labels must be 1D with length matching x's batch dim; "
            f"got x.shape={tuple(x.shape)}, labels.shape={tuple(labels.shape)}"
        )
    H, W = x.shape[-2], x.shape[-1]
    means = []
    vars_ = []
    n_total = 0
    for c in torch.unique(labels).tolist():
        mask = labels == c
        n_samples = int(mask.sum().item())
        n_elements = n_samples * H * W
        if n_elements < 2:
            continue
        x_c = x[mask]                                  # (n_samples, C, H, W)
        m = x_c.mean(dim=(0, 2, 3))                    # (C,)
        v = x_c.var(dim=(0, 2, 3), unbiased=False)     # (C,)
        means.append(m)
        vars_.append(v)
        n_total += n_elements
    if not means:
        # No class qualified. Fall back to standard batch stats so the model
        # still trains. The downstream caller still updates running stats.
        m = x.mean(dim=(0, 2, 3))
        v = x.var(dim=(0, 2, 3), unbiased=False)
        return m, v, x.shape[0] * H * W
    balanced_mean = torch.stack(means).mean(dim=0)
    balanced_var = torch.stack(vars_).mean(dim=0)
    return balanced_mean, balanced_var, n_total


class ClassBalancedLIBatchNorm2d(LIBatchNorm2d):
    """Drop-in subclass of LIBatchNorm2d. The integrated pathway uses class-
    balanced batch statistics during training when labels are stashed via
    ``stash_labels(...)``. Per-stream pathways and eval mode behavior are
    inherited unchanged from the parent class.

    The integrated pathway is identified inside ``_forward_single_pathway``
    by a buffer-identity check (`running_mean is self.integrated_running_mean`),
    so this override is a localized swap with no changes to the parent's
    forward orchestration (input checks, num_batches_tracked, exp_avg_factor,
    per-stream loop, modality-dropout active-idx handling).
    """

    def _forward_single_pathway(
        self,
        input: Tensor,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        exponential_average_factor: float,
        apply_relu: bool = False,
    ) -> Tensor:
        # Identify the integrated pathway via buffer identity. Per-stream
        # calls fall through to the parent's standard implementation.
        is_integrated = (
            running_mean is self.integrated_running_mean
            and running_var is self.integrated_running_var
        )
        labels = _peek_labels()
        active_cb = (
            self.training
            and self.track_running_stats
            and is_integrated
            and labels is not None
        )
        if not active_cb:
            return super()._forward_single_pathway(
                input, running_mean, running_var, weight, bias,
                exponential_average_factor, apply_relu,
            )

        # Class-balanced batch stats for the integrated pathway.
        # Integrated has no modality-dropout slicing → labels align row-by-row.
        if labels.shape[0] != input.shape[0]:
            # Unexpected mismatch — fall back to standard rather than crash.
            return super()._forward_single_pathway(
                input, running_mean, running_var, weight, bias,
                exponential_average_factor, apply_relu,
            )

        m, v, n_total = _class_balanced_batch_stats(input, labels)

        # Manual normalize-and-affine (replaces F.batch_norm).
        m_b = m.view(1, -1, 1, 1)
        v_b = v.view(1, -1, 1, 1)
        out = (input - m_b) / (v_b + self.eps).sqrt()
        if weight is not None:
            out = out * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

        # Running-stats update (mirrors standard BN's exponential moving avg
        # with Bessel correction on the variance estimate).
        if (
            running_mean is not None
            and running_var is not None
            and exponential_average_factor > 0.0
        ):
            with torch.no_grad():
                running_mean.mul_(1.0 - exponential_average_factor).add_(
                    m.detach(), alpha=exponential_average_factor
                )
                if n_total > 1:
                    var_unbiased = v.detach() * n_total / (n_total - 1)
                else:
                    var_unbiased = v.detach()
                running_var.mul_(1.0 - exponential_average_factor).add_(
                    var_unbiased, alpha=exponential_average_factor
                )

        if apply_relu:
            out = F.relu(out, inplace=True)
        return out


def convert_to_class_balanced_bn(model: nn.Module) -> int:
    """Replace every LIBatchNorm2d in ``model`` with ClassBalancedLIBatchNorm2d
    via in-place ``__class__`` rebinding. Preserves all parameters, buffers,
    and registered hooks — the operation is semantically a no-op until labels
    are stashed.

    Returns the number of modules converted. Idempotent: calling twice after
    the first call returns 0 (already-converted modules are skipped because
    the type check is exact, not isinstance).
    """
    count = 0
    for m in model.modules():
        # Use exact-type comparison so already-converted instances aren't
        # double-wrapped if this is called twice.
        if type(m) is LIBatchNorm2d:
            m.__class__ = ClassBalancedLIBatchNorm2d
            count += 1
    return count
