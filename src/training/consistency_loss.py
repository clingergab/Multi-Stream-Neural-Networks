"""
Symmetric KL consistency loss for two-view distribution-shift invariance.

Reference: Sajjadi et al. "Regularization With Stochastic Transformations and
Perturbations for Deep Semi-Supervised Learning" (NeurIPS 2016); also widely
used in FixMatch- and Mean Teacher-style methods. Forces the model to predict
the same softmax distribution for two stochastically augmented views of the
same input — when the augmentation pipeline simulates the train/test drift
axes (sensor variation, illumination, modality availability), this bakes
invariance to those axes into the features.

Loss form:
    p_a = softmax(logits_a / T)
    p_b = softmax(logits_b / T)
    KL_sym = 0.5 * (KL(p_a || p_b) + KL(p_b || p_a))

The symmetric form removes the choice of "which view is the target" — the
right call when the two views are produced by the same stochastic pipeline
(no teacher/student asymmetry).

Numerical stability:
    softmax/log_softmax over fp16 logits can underflow or NaN for sharp
    distributions. The loss internally promotes inputs to fp32 and disables
    autocast for the KL computation. Callers may invoke this from inside an
    autocast region without further wrapping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Symmetric KL divergence between two logit distributions, fp32-promoted.

    Args:
        logits_a: Logits from view A, shape ``[B, K]``.
        logits_b: Logits from view B, shape ``[B, K]``. Must be on the same
            device and have the same shape as ``logits_a``.
        temperature: Softmax temperature ``T``. ``T = 1.0`` is the standard
            softmax. Higher ``T`` (e.g. 2-4) softens the distributions and
            shrinks the KL gradient magnitude — common in knowledge-
            distillation and consistency-regularization literature.

    Returns:
        Scalar tensor: ``0.5 * (KL(p_a || p_b) + KL(p_b || p_a))``. Reduction
        is ``batchmean`` per ``F.kl_div`` convention.
    """
    with torch.amp.autocast(device_type=logits_a.device.type, enabled=False):
        a = logits_a.float() / temperature
        b = logits_b.float() / temperature
        log_p_a = F.log_softmax(a, dim=-1)
        log_p_b = F.log_softmax(b, dim=-1)
        p_a = log_p_a.exp()
        p_b = log_p_b.exp()
        kl_ab = F.kl_div(log_p_b, p_a, reduction='batchmean')
        kl_ba = F.kl_div(log_p_a, p_b, reduction='batchmean')
        return 0.5 * (kl_ab + kl_ba)
