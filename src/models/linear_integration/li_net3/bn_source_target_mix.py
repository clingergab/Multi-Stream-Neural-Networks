"""Source-target BN mixing for inference-time domain adaptation.

Motivation
----------
Cell 19.16 (AdaBN) showed +4.95pp on REP-drift classes by replacing running
stats with batch stats at inference, but with a -1.31pp top-line acc cost
because batch stats hurt majority classes whose running stats were already
well-fit. This module gives you the knob between those two extremes:

    effective_mean = α · running_mean + (1−α) · batch_mean
    effective_var  = α · running_var  + (1−α) · batch_var

    α=1.0  →  pure running stats     (standard eval; no AdaBN benefit)
    α=0.0  →  pure batch stats        (AdaBN; full benefit + full cost)
    α=0.7  →  mostly running, slight batch correction
    α=0.3  →  mostly batch, slight running stabilization

Tuning α on hpo_val finds the sweet spot where REP-drift classes gain
without majority-class regression. Pure inference-time, no retraining.

Scope
-----
Patches every ``LIBatchNorm2d`` in the model. Both per-stream and
integrated pathways are affected — at eval, modality dropout is off and
no active_idx slicing happens, so labels-aligned batch stats are
straightforward. The patch is via a context manager that restores the
original forward on exit; no permanent model modification.

Caveat (same as AdaBN)
----------------------
Requires test-time batches ≥ 32 for stable batch statistics, so this is
useful as a diagnostic + as a deployable choice IF you can serve in
batches. Single-sample inference can't use it.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conv import LIBatchNorm2d


_PATCHED_ATTR_ALPHA = "_bn_mix_alpha"
_PATCHED_ATTR_ORIGINAL = "_bn_mix_original_pathway"


def _blended_pathway(
    self: LIBatchNorm2d,
    input: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    exponential_average_factor: float,
    apply_relu: bool = False,
) -> Tensor:
    """Replacement ``_forward_single_pathway`` that uses α-blended running+batch
    stats. ``α`` is read from ``self._bn_mix_alpha`` (set by the context manager).
    Falls back to the original pathway when running stats are absent.
    """
    alpha = self._bn_mix_alpha
    if running_mean is None or running_var is None:
        return self._bn_mix_original_pathway(
            input, running_mean, running_var, weight, bias,
            exponential_average_factor, apply_relu,
        )

    # fp32 promotion (same lesson learned in CBBN — manual normalization must
    # match F.batch_norm's internal fp32 stat computation under autocast).
    in_dtype = input.dtype
    x = input.float() if in_dtype != torch.float32 else input

    # Per-channel batch stats. For LIBatchNorm2d, input is always (N, C, H, W).
    batch_mean = x.mean(dim=(0, 2, 3))
    batch_var = x.var(dim=(0, 2, 3), unbiased=False)

    # Blend with running stats (always fp32 buffers).
    blended_mean = alpha * running_mean.float() + (1.0 - alpha) * batch_mean
    blended_var = alpha * running_var.float() + (1.0 - alpha) * batch_var

    # Manual normalize-and-affine (replaces F.batch_norm, fp32 math).
    m_b = blended_mean.view(1, -1, 1, 1)
    v_b = blended_var.view(1, -1, 1, 1)
    out = (x - m_b) / (v_b + self.eps).sqrt()
    if weight is not None:
        out = out * weight.view(1, -1, 1, 1).float() + bias.view(1, -1, 1, 1).float()

    if in_dtype != torch.float32:
        out = out.to(in_dtype)
    if apply_relu:
        out = F.relu(out, inplace=True)
    return out


@contextlib.contextmanager
def bn_source_target_mix(model: nn.Module, alpha: float):
    """Context manager: each LIBatchNorm2d in ``model`` uses α-blended running+
    batch statistics during the wrapped forward. Restored on exit.

    Args:
        model: Any module containing LIBatchNorm2d submodules.
        alpha: Mix coefficient in [0, 1]. 1=pure running (eval baseline),
            0=pure batch (AdaBN), in-between values blend.

    Behavior:
        - Sets ``model.eval()`` so dropout/MD/etc are off (BN ONLY is affected
          by the alpha blend; everything else is normal eval).
        - Walks ``model.modules()`` and patches each LIBatchNorm2d's
          ``_forward_single_pathway`` to use blended stats.
        - On exit, restores every patched module's original method exactly.

    Idempotency:
        Re-entering the context with the same model and a different α just
        rebinds α via the per-instance attribute; the patch is preserved
        across re-entries (clean restore happens on the outermost exit).

    Raises:
        ValueError: if alpha not in [0, 1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    model.eval()
    patched_modules: list[LIBatchNorm2d] = []
    for m in model.modules():
        if isinstance(m, LIBatchNorm2d):
            # Stash original (only on first entry; nested contexts reuse it)
            if not hasattr(m, _PATCHED_ATTR_ORIGINAL):
                setattr(m, _PATCHED_ATTR_ORIGINAL, m._forward_single_pathway)
                # Bind the replacement method to the instance
                m._forward_single_pathway = _blended_pathway.__get__(m, type(m))
                patched_modules.append(m)
            setattr(m, _PATCHED_ATTR_ALPHA, alpha)

    try:
        yield
    finally:
        for m in patched_modules:
            # Restore — only modules we patched on THIS entry
            m._forward_single_pathway = getattr(m, _PATCHED_ATTR_ORIGINAL)
            delattr(m, _PATCHED_ATTR_ORIGINAL)
            if hasattr(m, _PATCHED_ATTR_ALPHA):
                delattr(m, _PATCHED_ATTR_ALPHA)
