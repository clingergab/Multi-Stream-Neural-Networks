"""Unit tests for src.training.sam.

Covers:
- SAM.first_step perturbation correctness (direction, magnitude == rho)
- zero_grad behavior between steps
- SAM.second_step restoration and base-optimizer step
- disable_running_stats / enable_running_stats on nn.BatchNorm2d AND
  LIBatchNorm2d (dual-BN coverage).
"""

import pytest
import torch
import torch.nn as nn

from src.models.linear_integration.li_net3.conv import LIBatchNorm2d
from training.sam import SAM, disable_running_stats, enable_running_stats


class TestSAMConstructor:
    def test_rho_must_be_positive(self):
        params = [torch.nn.Parameter(torch.randn(4))]
        opt = torch.optim.SGD(params, lr=0.1)
        with pytest.raises(ValueError, match="rho must be > 0"):
            SAM(opt, rho=0.0)
        with pytest.raises(ValueError, match="rho must be > 0"):
            SAM(opt, rho=-0.1)

    def test_param_groups_proxy(self):
        params = [torch.nn.Parameter(torch.randn(4))]
        opt = torch.optim.SGD(params, lr=0.1)
        sam = SAM(opt, rho=0.05)
        # Identity proxy — same underlying list
        assert sam.param_groups is opt.param_groups


class TestSAMFirstStep:
    def test_perturbation_magnitude_equals_rho(self):
        """After first_step, ||w - w_0||_2 over all params should equal rho."""
        rho = 0.05
        p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        opt = torch.optim.SGD([p1, p2], lr=0.1)
        sam = SAM(opt, rho=rho)

        # Populate gradients
        w0 = [p1.detach().clone(), p2.detach().clone()]
        p1.grad = torch.tensor([0.1, 0.2])
        p2.grad = torch.tensor([[0.3, -0.1], [0.0, 0.5]])

        sam.first_step(zero_grad=False)

        # Combined L2 displacement across all params should equal rho
        disp = torch.cat([(p1 - w0[0]).flatten(), (p2 - w0[1]).flatten()])
        assert disp.norm(p=2).item() == pytest.approx(rho, rel=1e-5)

    def test_perturbation_direction_matches_gradient(self):
        rho = 0.05
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        opt = torch.optim.SGD([p], lr=0.1)
        sam = SAM(opt, rho=rho)
        w0 = p.detach().clone()
        g = torch.tensor([0.5, -0.5, 1.0])
        p.grad = g.clone()

        sam.first_step(zero_grad=False)

        # delta should point in the direction of grad
        delta = p - w0
        # normalize both, check cosine similarity = 1
        cos = torch.nn.functional.cosine_similarity(
            delta.unsqueeze(0), g.unsqueeze(0)
        ).item()
        assert cos == pytest.approx(1.0, rel=1e-5)

    def test_zero_grad_true_clears_gradients(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = torch.optim.SGD([p], lr=0.1)
        sam = SAM(opt, rho=0.05)
        p.grad = torch.tensor([0.1, 0.2])
        sam.first_step(zero_grad=True)
        assert p.grad is None or p.grad.abs().sum().item() == 0.0

    def test_zero_grad_false_preserves_gradients(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = torch.optim.SGD([p], lr=0.1)
        sam = SAM(opt, rho=0.05)
        p.grad = torch.tensor([0.1, 0.2])
        sam.first_step(zero_grad=False)
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0


class TestSAMSecondStep:
    def test_second_step_restores_weights_and_applies_base_step(self):
        """t0 = initial, t1 = after first_step, t2 = after second_step.
        Expect t2 = t0 - lr * grad2."""
        lr = 0.1
        rho = 0.05
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        opt = torch.optim.SGD([p], lr=lr)
        sam = SAM(opt, rho=rho)

        t0 = p.detach().clone()

        # First pass
        p.grad = torch.tensor([1.0, 0.0, 0.0])
        sam.first_step(zero_grad=True)
        # t1 = t0 + rho * [1,0,0] / 1 = t0 + [rho, 0, 0]
        t1 = p.detach().clone()
        assert torch.allclose(t1, t0 + torch.tensor([rho, 0.0, 0.0]))

        # Second pass gradient
        g2 = torch.tensor([0.2, 0.3, 0.4])
        p.grad = g2.clone()
        sam.second_step(zero_grad=True)

        # Expect t2 = t0 - lr * g2 (first restore to t0, then SGD step)
        t2 = p.detach().clone()
        assert torch.allclose(t2, t0 - lr * g2, atol=1e-6)

    def test_second_step_zero_grad(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = torch.optim.SGD([p], lr=0.1)
        sam = SAM(opt, rho=0.05)
        p.grad = torch.tensor([1.0, 0.0])
        sam.first_step(zero_grad=True)
        p.grad = torch.tensor([0.5, 0.5])
        sam.second_step(zero_grad=True)
        assert p.grad is None or p.grad.abs().sum().item() == 0.0


class TestBNRunningStatsHelpers:
    def _make_mixed_bn_model(self):
        """Small toy model with both nn.BatchNorm2d and LIBatchNorm2d."""
        model = nn.ModuleDict({
            "vanilla_bn": nn.BatchNorm2d(8),
            "li_bn": LIBatchNorm2d(stream_num_features=[8, 8], integrated_num_features=8),
        })
        model.train()
        return model

    def test_disable_running_stats_sets_momentum_zero_on_bn(self):
        model = self._make_mixed_bn_model()
        orig_vanilla_mom = model["vanilla_bn"].momentum
        orig_li_mom = model["li_bn"].momentum
        disable_running_stats(model)
        assert model["vanilla_bn"].momentum == 0.0
        assert model["li_bn"].momentum == 0.0
        # Original values stashed
        assert model["vanilla_bn"]._sam_backup_momentum == orig_vanilla_mom
        assert model["li_bn"]._sam_backup_momentum == orig_li_mom

    def test_enable_running_stats_restores_momentum(self):
        model = self._make_mixed_bn_model()
        orig_vanilla_mom = model["vanilla_bn"].momentum
        orig_li_mom = model["li_bn"].momentum
        disable_running_stats(model)
        enable_running_stats(model)
        assert model["vanilla_bn"].momentum == orig_vanilla_mom
        assert model["li_bn"].momentum == orig_li_mom
        # Backup attribute removed
        assert not hasattr(model["vanilla_bn"], "_sam_backup_momentum")
        assert not hasattr(model["li_bn"], "_sam_backup_momentum")

    def test_enable_without_disable_is_safe_noop(self):
        model = self._make_mixed_bn_model()
        orig = model["vanilla_bn"].momentum
        enable_running_stats(model)  # should not error
        assert model["vanilla_bn"].momentum == orig

    def test_disable_running_stats_actually_prevents_updates_vanilla_bn(self):
        """Critical end-to-end behavior test: with running stats disabled, forward
        in training mode should NOT change running_mean/running_var."""
        bn = nn.BatchNorm2d(8)
        bn.train()

        x = torch.randn(4, 8, 16, 16)
        # Baseline: running stats DO update
        rm_before = bn.running_mean.clone()
        _ = bn(x)
        rm_after = bn.running_mean.clone()
        assert (rm_after - rm_before).abs().sum().item() > 1e-6

        # With disable_running_stats, running stats should NOT update
        disable_running_stats(bn)
        rm_before = bn.running_mean.clone()
        _ = bn(x)
        rm_after = bn.running_mean.clone()
        assert (rm_after - rm_before).abs().sum().item() < 1e-8

        # After re-enable, updates resume
        enable_running_stats(bn)
        rm_before = bn.running_mean.clone()
        _ = bn(x)
        rm_after = bn.running_mean.clone()
        assert (rm_after - rm_before).abs().sum().item() > 1e-6

    def test_disable_running_stats_actually_prevents_updates_li_bn(self):
        """Same end-to-end test for LIBatchNorm2d."""
        bn = LIBatchNorm2d(stream_num_features=[8, 8], integrated_num_features=8)
        bn.train()

        stream_inputs = [torch.randn(4, 8, 16, 16), torch.randn(4, 8, 16, 16)]
        integrated_input = torch.randn(4, 8, 16, 16)

        # Baseline: running stats DO update
        rm0_before = bn.stream0_running_mean.clone()
        rmi_before = bn.integrated_running_mean.clone()
        _ = bn(stream_inputs, integrated_input)
        rm0_after = bn.stream0_running_mean.clone()
        rmi_after = bn.integrated_running_mean.clone()
        assert (rm0_after - rm0_before).abs().sum().item() > 1e-6
        assert (rmi_after - rmi_before).abs().sum().item() > 1e-6

        # With disable_running_stats, running stats should NOT update
        disable_running_stats(bn)
        rm0_before = bn.stream0_running_mean.clone()
        rmi_before = bn.integrated_running_mean.clone()
        _ = bn(stream_inputs, integrated_input)
        rm0_after = bn.stream0_running_mean.clone()
        rmi_after = bn.integrated_running_mean.clone()
        assert (rm0_after - rm0_before).abs().sum().item() < 1e-8
        assert (rmi_after - rmi_before).abs().sum().item() < 1e-8

        # After re-enable, updates resume
        enable_running_stats(bn)
        rm0_before = bn.stream0_running_mean.clone()
        _ = bn(stream_inputs, integrated_input)
        rm0_after = bn.stream0_running_mean.clone()
        assert (rm0_after - rm0_before).abs().sum().item() > 1e-6
