"""Unit tests for src.training.mixup."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from training.mixup import mixup_batch, mixup_loss


class TestMixupBatch:
    def test_output_shapes_match_inputs(self):
        rgb = torch.randn(8, 3, 16, 16)
        depth = torch.randn(8, 1, 16, 16)
        labels = torch.randint(0, 5, (8,))
        mixed, la, lb, lam = mixup_batch([rgb, depth], labels, alpha=0.2)
        assert len(mixed) == 2
        assert mixed[0].shape == rgb.shape
        assert mixed[1].shape == depth.shape
        assert la.shape == labels.shape
        assert lb.shape == labels.shape
        assert 0.0 <= lam <= 1.0

    def test_alpha_zero_disables_mixup(self):
        """alpha <= 0 returns inputs unchanged with lam=1.0."""
        rgb = torch.randn(8, 3, 16, 16)
        depth = torch.randn(8, 1, 16, 16)
        labels = torch.randint(0, 5, (8,))
        mixed, la, lb, lam = mixup_batch([rgb, depth], labels, alpha=0.0)
        assert lam == 1.0
        assert torch.equal(mixed[0], rgb)
        assert torch.equal(mixed[1], depth)
        assert torch.equal(la, labels)
        assert torch.equal(lb, labels)

    def test_mixing_formula_applied_correctly(self):
        """mixed[k] == lam * streams[k] + (1-lam) * streams[k][perm] — same perm for every stream."""
        torch.manual_seed(0)
        rgb = torch.randn(4, 3, 8, 8)
        depth = torch.randn(4, 1, 8, 8)
        labels = torch.tensor([0, 1, 2, 3])

        # Force a deterministic lam and deterministic permutation
        fake_perm = torch.tensor([2, 0, 3, 1])

        with patch("training.mixup.np.random.beta", return_value=0.3), \
             patch("training.mixup.torch.randperm", return_value=fake_perm):
            mixed, la, lb, lam = mixup_batch([rgb, depth], labels, alpha=0.2)

        assert lam == pytest.approx(0.3)
        expected_rgb = 0.3 * rgb + 0.7 * rgb[fake_perm]
        expected_depth = 0.3 * depth + 0.7 * depth[fake_perm]
        assert torch.allclose(mixed[0], expected_rgb, atol=1e-6)
        assert torch.allclose(mixed[1], expected_depth, atol=1e-6)
        assert torch.equal(la, labels)
        assert torch.equal(lb, labels[fake_perm])

    def test_same_permutation_applied_to_all_streams(self):
        """Both streams must be permuted with the SAME permutation (preserves RGB-Depth pairing).

        Directly verifies the mixing formula for each stream using a fixed perm
        via patching, avoiding fp32 back-solve precision issues.
        """
        rgb = torch.randn(6, 3, 8, 8)
        depth = torch.randn(6, 1, 8, 8)
        labels = torch.arange(6)

        fake_perm = torch.tensor([3, 5, 0, 2, 4, 1])
        for fake_lam in [0.2, 0.4, 0.6, 0.8]:
            with patch("training.mixup.np.random.beta", return_value=fake_lam), \
                 patch("training.mixup.torch.randperm", return_value=fake_perm):
                mixed, la, lb, lam = mixup_batch([rgb, depth], labels, alpha=0.5)

            # Same lam, same perm ⇒ mixed = lam * x + (1-lam) * x[perm] for each stream
            expected_rgb = fake_lam * rgb + (1.0 - fake_lam) * rgb[fake_perm]
            expected_depth = fake_lam * depth + (1.0 - fake_lam) * depth[fake_perm]
            assert torch.allclose(mixed[0], expected_rgb, atol=1e-6)
            assert torch.allclose(mixed[1], expected_depth, atol=1e-6)
            # Labels use the same permutation
            assert torch.equal(lb, labels[fake_perm])


class TestMixupLoss:
    def test_lambda_equals_one_matches_base_criterion(self):
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(8, 10)
        labels_a = torch.randint(0, 10, (8,))
        labels_b = torch.randint(0, 10, (8,))
        expected = criterion(logits, labels_a)
        got = mixup_loss(criterion, logits, labels_a, labels_b, lam=1.0)
        assert torch.allclose(got, expected)

    def test_lambda_equals_zero_matches_criterion_on_labels_b(self):
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(8, 10)
        labels_a = torch.randint(0, 10, (8,))
        labels_b = torch.randint(0, 10, (8,))
        expected = criterion(logits, labels_b)
        got = mixup_loss(criterion, logits, labels_a, labels_b, lam=0.0)
        assert torch.allclose(got, expected)

    def test_linear_blend_of_two_losses(self):
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(8, 10)
        labels_a = torch.randint(0, 10, (8,))
        labels_b = torch.randint(0, 10, (8,))
        lam = 0.3
        expected = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        got = mixup_loss(criterion, logits, labels_a, labels_b, lam=lam)
        assert torch.allclose(got, expected)

    def test_works_with_label_smoothing(self):
        """Mixup loss composes with label-smoothing CE (criterion just runs twice)."""
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        logits = torch.randn(8, 10)
        labels_a = torch.randint(0, 10, (8,))
        labels_b = torch.randint(0, 10, (8,))
        lam = 0.4
        expected = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        got = mixup_loss(criterion, logits, labels_a, labels_b, lam=lam)
        assert torch.allclose(got, expected)
