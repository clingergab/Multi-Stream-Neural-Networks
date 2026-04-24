"""
MixUp utilities for multi-stream classification.

Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018).

Per-batch mixup: draw λ ~ Beta(α, α), construct a mixed batch by linearly
combining each sample with a randomly permuted partner. For multi-stream
models (RGB + Depth), the SAME λ and SAME permutation are applied to every
stream so the RGB-Depth correspondence is preserved.

Interaction with label smoothing:
    With smoothing ``s`` and mixup ``λ``, the effective target for a sample is
        (1-s) · (λ · onehot(y_a) + (1-λ) · onehot(y_b)) + s/K
    This is mathematically correct but stacks two softening effects. Moderate
    smoothing (s ≤ 0.08) composes fine with typical mixup (α=0.2–0.4).
    Aggressive smoothing (s ≥ 0.15) combined with mixup can under-train the
    classifier; s=0.12 is borderline. Document any ablation that stacks both.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def mixup_batch(
    streams: Sequence[torch.Tensor],
    labels: torch.Tensor,
    alpha: float,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a multi-stream batch.

    Args:
        streams: List of N input tensors, each of shape ``[B, C_i, H, W]``.
            For LiNet3 this is ``[rgb, depth]`` with shapes ``[B, 3, H, W]``
            and ``[B, 1, H, W]``.
        labels: Class labels of shape ``[B]`` (int64).
        alpha: Beta distribution parameter. If ``<= 0``, no mixup is applied
            (returns inputs unchanged with ``lam = 1.0`` and ``labels_b =
            labels``).

    Returns:
        mixed_streams: List of N tensors, each equal to
            ``lam * streams[k] + (1 - lam) * streams[k][perm]``.
            Uses the SAME lam and SAME permutation for every stream.
        labels_a: Original labels (``labels``).
        labels_b: Permuted labels (``labels[perm]``).
        lam: The mixing coefficient ``λ ∈ [0, 1]``.
    """
    if alpha <= 0:
        # Mixup disabled — pass through unchanged
        return list(streams), labels, labels, 1.0

    batch_size = streams[0].shape[0]
    device = streams[0].device

    # Draw lam; draw on CPU to avoid device-specific numpy/cuda interaction
    lam = float(np.random.beta(alpha, alpha))

    perm = torch.randperm(batch_size, device=device)

    mixed_streams = [lam * s + (1.0 - lam) * s[perm] for s in streams]
    labels_a = labels
    labels_b = labels[perm]
    return mixed_streams, labels_a, labels_b, lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the mixup loss: ``λ · L(logits, y_a) + (1 - λ) · L(logits, y_b)``.

    Compatible with any CE-style criterion, including
    ``nn.CrossEntropyLoss(label_smoothing=...)``. The criterion is called
    twice and the scalar results are blended.

    When ``lam == 1.0`` (mixup disabled), reduces to ``criterion(logits, labels_a)``.
    """
    if lam >= 1.0:
        return criterion(logits, labels_a)
    if lam <= 0.0:
        return criterion(logits, labels_b)
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
