"""Tests for DecayingCosineAnnealingWarmRestarts with fractional T_mult."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from src.training.schedulers import DecayingCosineAnnealingWarmRestarts


def get_lrs_and_restarts(scheduler, optimizer, epochs):
    """
    Helper to simulate training and collect LRs and restart points.

    Pattern:
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]['lr']  # LR for this epoch
            train(...)
            scheduler.step()  # Update for next epoch
    """
    lrs = []
    restarts = []

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

        # Detect restart (LR jumps up significantly)
        if epoch > 0 and lr > lrs[epoch-1] * 1.5:
            restarts.append(epoch)

        scheduler.step()

    return lrs, restarts


class TestFractionalTMult:
    """Test fractional T_mult values."""

    def test_tmult_1_5(self):
        """Test T_mult=1.5 produces correct cycle lengths."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1.5, eta_min=0.0, restart_decay=0.9
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 100)

        # Expected cycle lengths: 10, floor(10*1.5)=15, floor(15*1.5)=22, floor(22*1.5)=33
        # Expected restarts at: 10, 10+15=25, 25+22=47, 47+33=80
        expected_restarts = [10, 25, 47, 80]

        assert restarts[:4] == expected_restarts, \
            f"Expected restarts at {expected_restarts}, got {restarts[:4]}"

    def test_tmult_1_2(self):
        """Test T_mult=1.2 produces correct cycle lengths."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1.2, eta_min=0.0, restart_decay=0.8
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 100)

        # Cycle lengths: 10, floor(10*1.2)=12, floor(12*1.2)=14, floor(14*1.2)=16, floor(16*1.2)=19
        # Restarts: 10, 22, 36, 52, 71
        expected_restarts = [10, 22, 36, 52, 71]

        assert restarts[:5] == expected_restarts, \
            f"Expected restarts at {expected_restarts[:5]}, got {restarts[:5]}"

    def test_tmult_2_5(self):
        """Test T_mult=2.5 produces correct cycle lengths."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2.5, eta_min=0.0, restart_decay=0.9
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 100)

        # Cycle lengths: 5, floor(5*2.5)=12, floor(12*2.5)=30, floor(30*2.5)=75
        # Restarts: 5, 17, 47
        expected_restarts = [5, 17, 47]

        assert restarts[:3] == expected_restarts, \
            f"Expected restarts at {expected_restarts}, got {restarts[:3]}"

    def test_tmult_1_0_exact(self):
        """Test T_mult=1.0 (exact float) works like T_mult=1."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1.0, eta_min=0.0, restart_decay=0.8
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 50)

        # Should have constant cycle length of 10
        expected_restarts = [10, 20, 30, 40]

        assert restarts == expected_restarts, \
            f"Expected restarts at {expected_restarts}, got {restarts}"

    def test_decay_with_fractional_tmult(self):
        """Test that decay still works correctly with fractional T_mult."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1.5, eta_min=0.0, restart_decay=0.8
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 60)

        # Peak LRs at restarts should decay by 0.8 each time
        peaks = [lrs[r] for r in restarts]
        expected_peaks = [0.08, 0.064, 0.0512]

        for i, (actual, expected) in enumerate(zip(peaks, expected_peaks)):
            assert abs(actual - expected) < 1e-4, \
                f"Peak {i+1}: expected {expected}, got {actual}"

    def test_state_dict_with_fractional_tmult(self):
        """Test state dict save/load with fractional T_mult."""
        model = nn.Linear(10, 2)
        optimizer1 = SGD(model.parameters(), lr=0.1)
        scheduler1 = DecayingCosineAnnealingWarmRestarts(
            optimizer1, T_0=10, T_mult=1.5, eta_min=0.001, restart_decay=0.8
        )

        # Run for 30 epochs
        for _ in range(30):
            scheduler1.step()

        state = scheduler1.state_dict()
        lr_before = optimizer1.param_groups[0]['lr']

        # Create new scheduler and load state
        optimizer2 = SGD(model.parameters(), lr=0.1)
        scheduler2 = DecayingCosineAnnealingWarmRestarts(
            optimizer2, T_0=10, T_mult=1.5, eta_min=0.001, restart_decay=0.8
        )
        scheduler2.load_state_dict(state)

        lr_after = optimizer2.param_groups[0]['lr']

        assert abs(lr_before - lr_after) < 1e-10, \
            f"LR mismatch: {lr_before} vs {lr_after}"

        # Continue both and verify they stay in sync
        for _ in range(10):
            scheduler1.step()
            scheduler2.step()

            lr1 = optimizer1.param_groups[0]['lr']
            lr2 = optimizer2.param_groups[0]['lr']

            assert abs(lr1 - lr2) < 1e-10, f"LRs diverged: {lr1} vs {lr2}"


class TestBackwardCompatibility:
    """Test that integer T_mult still works correctly."""

    def test_integer_tmult_2(self):
        """Test integer T_mult=2 works as before."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.001, restart_decay=0.8
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 80)

        # Expected: 10, 30, 70
        expected_restarts = [10, 30, 70]

        assert restarts == expected_restarts, \
            f"Expected restarts at {expected_restarts}, got {restarts}"

        # Check peak LRs
        peaks = [lrs[r] for r in restarts]
        expected_peaks = [0.08, 0.064, 0.0512]

        for i, (actual, expected) in enumerate(zip(peaks, expected_peaks)):
            assert abs(actual - expected) < 1e-5, \
                f"Peak {i+1}: expected {expected}, got {actual}"

    def test_integer_tmult_1(self):
        """Test integer T_mult=1 (constant cycles)."""
        model = nn.Linear(10, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = DecayingCosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=0.0, restart_decay=0.8
        )

        lrs, restarts = get_lrs_and_restarts(scheduler, optimizer, 50)

        expected_restarts = [10, 20, 30, 40]

        assert restarts == expected_restarts, \
            f"Expected restarts at {expected_restarts}, got {restarts}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
