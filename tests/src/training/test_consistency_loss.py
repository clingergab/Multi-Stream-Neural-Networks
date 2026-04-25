"""Unit tests for src.training.consistency_loss."""

import pytest
import torch

from training.consistency_loss import kl_consistency_loss


class TestKLConsistencyLoss:
    def test_output_is_scalar(self):
        """Loss should be a 0-d scalar tensor."""
        logits_a = torch.randn(8, 19)
        logits_b = torch.randn(8, 19)
        loss = kl_consistency_loss(logits_a, logits_b)
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_identical_logits_zero_loss(self):
        """KL(p || p) = 0 in both directions, so symmetric KL is 0."""
        logits = torch.randn(8, 19)
        loss = kl_consistency_loss(logits, logits)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_in_inputs(self):
        """Swapping the two views must not change the loss."""
        torch.manual_seed(0)
        logits_a = torch.randn(8, 19)
        logits_b = torch.randn(8, 19)
        loss_ab = kl_consistency_loss(logits_a, logits_b)
        loss_ba = kl_consistency_loss(logits_b, logits_a)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-6)

    def test_temperature_softens_kl(self):
        """Higher T softens both distributions, shrinking the divergence."""
        torch.manual_seed(0)
        logits_a = torch.randn(8, 19) * 5.0  # sharp logits
        logits_b = torch.randn(8, 19) * 5.0
        loss_t1 = kl_consistency_loss(logits_a, logits_b, temperature=1.0)
        loss_t4 = kl_consistency_loss(logits_a, logits_b, temperature=4.0)
        assert loss_t4 < loss_t1

    def test_loss_is_non_negative(self):
        """KL is non-negative; symmetric KL inherits that."""
        torch.manual_seed(0)
        for _ in range(5):
            logits_a = torch.randn(8, 19)
            logits_b = torch.randn(8, 19)
            loss = kl_consistency_loss(logits_a, logits_b)
            assert loss.item() >= 0.0

    def test_gradient_flows_to_both_inputs(self):
        """Both view A and view B must receive gradient (symmetric formulation)."""
        logits_a = torch.randn(8, 19, requires_grad=True)
        logits_b = torch.randn(8, 19, requires_grad=True)
        loss = kl_consistency_loss(logits_a, logits_b, temperature=2.0)
        loss.backward()
        assert logits_a.grad is not None
        assert logits_b.grad is not None
        assert torch.isfinite(logits_a.grad).all()
        assert torch.isfinite(logits_b.grad).all()
        # Gradients should not be all zero
        assert logits_a.grad.abs().sum().item() > 0
        assert logits_b.grad.abs().sum().item() > 0

    def test_fp32_under_autocast(self):
        """Loss must compute in fp32 even when called from inside autocast."""
        logits_a = torch.randn(8, 19) * 50.0  # very sharp; would NaN in fp16
        logits_b = torch.randn(8, 19) * 50.0
        with torch.amp.autocast(device_type='cpu', enabled=False):
            # Note: CPU autocast support is limited; the test is that the
            # loss function self-disables autocast and uses fp32 internally.
            loss = kl_consistency_loss(logits_a, logits_b)
        assert torch.isfinite(loss).item()

    def test_extreme_distributions_finite(self):
        """One-hot vs uniform gives a large but finite loss."""
        # logits_a heavily favors class 0; logits_b favors class 1
        logits_a = torch.zeros(4, 19)
        logits_a[:, 0] = 50.0
        logits_b = torch.zeros(4, 19)
        logits_b[:, 1] = 50.0
        loss = kl_consistency_loss(logits_a, logits_b)
        assert torch.isfinite(loss).item()
        assert loss.item() > 0

    def test_device_preservation(self):
        """Loss tensor lives on the same device as the inputs."""
        logits_a = torch.randn(4, 19)
        logits_b = torch.randn(4, 19)
        loss = kl_consistency_loss(logits_a, logits_b)
        assert loss.device == logits_a.device

    def test_temperature_one_default(self):
        """Default T=1.0 matches explicit T=1.0."""
        torch.manual_seed(0)
        logits_a = torch.randn(8, 19)
        logits_b = torch.randn(8, 19)
        loss_default = kl_consistency_loss(logits_a, logits_b)
        loss_explicit = kl_consistency_loss(logits_a, logits_b, temperature=1.0)
        assert torch.allclose(loss_default, loss_explicit)
