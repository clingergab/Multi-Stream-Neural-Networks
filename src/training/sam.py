"""
Sharpness-Aware Minimization (SAM) — Foret et al., ICLR 2021.

SAM takes two forward+backward passes per step:
    1. Forward/backward at current weights w → gradient g.
    2. Compute perturbation ε = ρ · g / ||g||; apply: w ← w + ε.
    3. Forward/backward at perturbed weights → gradient g'.
    4. Restore weights: w ← w - ε.
    5. Apply base optimizer using g': w ← w - lr · g'.

This implementation is a *plain wrapper*, NOT a ``torch.optim.Optimizer``
subclass. Inheriting from ``Optimizer`` creates ambiguity for LR schedulers
(which ``step()`` do they target?) and learning-rate-tracking utilities that
expect a concrete optimizer. By exposing ``base_optimizer`` as an attribute,
schedulers and other tooling wire to the base directly; SAM only adds
``first_step`` / ``second_step`` methods that the training loop calls.

BN running-stats handling:
    Each batch enters the forward pass twice under SAM, which would cause
    running statistics to update twice per batch (skewing them vs. a non-SAM
    baseline). Use ``disable_running_stats(model)`` before the second forward
    and ``enable_running_stats(model)`` after. Implementation: temporarily set
    each BN module's ``momentum`` attribute to 0 (exponential_average_factor
    becomes 0 → running stats are not updated). This works for both
    ``torch.nn.BatchNorm{1,2,3}d`` and ``src.models.linear_integration.
    li_net3.conv.LIBatchNorm2d`` (verified empirically: momentum=0 freezes
    running-stats updates without affecting dropout/affine params).

AMP + SAM:
    ``torch.amp.GradScaler.unscale_()`` can only be called once per iteration
    before ``step()`` / ``update()``. The two-pass SAM design does not compose
    cleanly with a single shared GradScaler. Recommended usage: when SAM is
    active, run training in fp32 (disable AMP). See ``li_net.py`` fit() for
    the concrete integration.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn


class SAM:
    """Sharpness-Aware Minimization wrapper around a base optimizer.

    Not a ``torch.optim.Optimizer`` subclass. Exposes:
        - ``base_optimizer``: the wrapped optimizer (pass THIS to LR schedulers)
        - ``rho``: perturbation radius
        - ``param_groups``: proxy to ``base_optimizer.param_groups`` (convenience
          only; intended for read-only inspection, not as an "optimizer").
        - ``first_step(zero_grad)``: perturb weights along gradient
        - ``second_step(zero_grad)``: restore weights and call ``base_optimizer.step()``
        - ``zero_grad(set_to_none)``: delegates to ``base_optimizer.zero_grad``

    Example (non-AMP):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SAM(optimizer, rho=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sam.base_optimizer, ...)

        for x, y in loader:
            # first forward + backward
            loss = criterion(model(x), y); loss.backward()
            sam.first_step(zero_grad=True)

            # second forward + backward with BN frozen
            disable_running_stats(model)
            loss2 = criterion(model(x), y); loss2.backward()
            sam.second_step(zero_grad=True)
            enable_running_stats(model)
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        # Per-parameter stash of the applied perturbation so second_step can undo it
        self._e_w: dict[int, torch.Tensor] = {}

    @property
    def param_groups(self):
        """Read-only proxy to the base optimizer's param_groups.

        Intended for inspection only (e.g., reading LR values for logging).
        Schedulers should wrap ``base_optimizer`` directly, not this property.
        """
        return self.base_optimizer.param_groups

    def _iter_params_with_grad(self) -> Iterator[torch.nn.Parameter]:
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    yield p

    def _grad_norm(self) -> torch.Tensor:
        """L2 norm of all gradients across all param groups."""
        grads = [p.grad.detach() for p in self._iter_params_with_grad()]
        if len(grads) == 0:
            return torch.tensor(0.0)
        # Compute on the device of the first gradient
        device = grads[0].device
        return torch.norm(
            torch.stack([g.norm(p=2).to(device) for g in grads]), p=2
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb: w ← w + ρ · g / ||g||.

        Call AFTER the first forward+backward pass. Must be followed by a
        second forward+backward and ``second_step``.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        self._e_w.clear()
        for p in self._iter_params_with_grad():
            e_w = p.grad * scale.to(p.device)
            p.add_(e_w)
            # Stash perturbation keyed by parameter id so second_step can reverse it
            self._e_w[id(p)] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore weights and apply base optimizer step using second-pass gradients.

        Call AFTER the second forward+backward pass.
        """
        # Restore weights (w ← w - e_w)
        for p in self._iter_params_with_grad():
            e_w = self._e_w.get(id(p))
            if e_w is not None:
                p.sub_(e_w)
        self._e_w.clear()
        # Apply base optimizer using gradients from the second forward
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# ---------------------------------------------------------------------------
# BN running-stats toggles for SAM's second forward pass
# ---------------------------------------------------------------------------

_BN_TYPES: tuple = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _is_bn_like(module: nn.Module) -> bool:
    """True for standard torch BN modules AND LIBatchNorm2d (same `momentum` API).

    Walks the class MRO to recognize subclasses of LIBatchNorm2d (e.g.
    ``ClassBalancedLIBatchNorm2d``). The previous name-string check
    (``type(module).__name__ == "LIBatchNorm2d"``) silently excluded
    subclasses, which caused SAM's second-pass running-stats freeze to skip
    them — corrupting their running stats with statistics computed at the
    adversarially-perturbed weights and breaking eval-mode forward.

    Duck-typed (MRO walk by name) rather than ``isinstance(..., LIBatchNorm2d)``
    to avoid a circular import: ``src.models.linear_integration.li_net3.__init__``
    imports ``li_net.py`` which imports back from ``src.training.sam``.
    """
    if isinstance(module, _BN_TYPES):
        return True
    return any(cls.__name__ == "LIBatchNorm2d" for cls in type(module).__mro__)


def disable_running_stats(model: nn.Module) -> None:
    """Freeze BN running statistics by setting ``momentum`` to 0 on all BN
    modules. Call before SAM's second forward pass.

    Stashes the original momentum on each BN module as ``_sam_backup_momentum``
    so ``enable_running_stats`` can restore it. Covers standard
    ``nn.BatchNorm{1,2,3}d`` / ``SyncBatchNorm`` and ``LIBatchNorm2d``.

    Affects running stats only. Affine params, dropout, and model.training
    are untouched (BN still uses mini-batch statistics for normalization, so
    gradients flow correctly).
    """
    for module in model.modules():
        if _is_bn_like(module):
            module._sam_backup_momentum = module.momentum
            module.momentum = 0.0


def enable_running_stats(model: nn.Module) -> None:
    """Restore BN ``momentum`` attributes stashed by ``disable_running_stats``.

    Safe to call even if ``disable_running_stats`` was not called (no-op for
    modules without a backup attribute). Call after SAM's second forward pass.
    """
    for module in model.modules():
        if _is_bn_like(module) and hasattr(module, "_sam_backup_momentum"):
            module.momentum = module._sam_backup_momentum
            del module._sam_backup_momentum
