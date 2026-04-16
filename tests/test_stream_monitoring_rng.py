"""
Test that stream_monitoring has zero side effects on training RNG state.

The bug: _evaluate_stream_contributions iterates a DataLoader backed by the
training dataset, whose __getitem__ fires random augmentations (RandomCrop,
flip, etc.), consuming the global numpy/torch RNG. This shifts subsequent
training epochs to get different random augmentations.

The fix: save and restore numpy + torch RNG state around the monitoring
_validate calls in _evaluate_stream_contributions.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = 32
NUM_CLASSES = 5
BATCH_SIZE = 4
NUM_TRAIN = 24
EPOCHS = 4


class AugmentedTupleDataset(Dataset):
    """
    Dataset that returns (stream1, stream2, label) tuples AND consumes
    global RNG in __getitem__ — mimicking SUNRGBDDataset's random
    augmentations (RandomCrop, flip, jitter, depth noise).
    """

    def __init__(self, num_samples, num_streams=2):
        self.num_samples = num_samples
        self.num_streams = num_streams
        # Fixed data — the augmentation noise is the only source of randomness
        self.streams = [
            torch.randn(num_samples, 3 if i == 0 else 1, IMAGE_SIZE, IMAGE_SIZE)
            for i in range(num_streams)
        ]
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        items = []
        for s in self.streams:
            tensor = s[idx].clone()
            # Simulate RandomCrop.get_params — consumes torch CPU RNG
            torch.randint(0, 5, (2,))
            # Simulate flip decision — consumes numpy RNG
            np.random.random()
            # Simulate color jitter decision — consumes numpy RNG
            np.random.random()
            # Simulate depth noise — consumes torch RNG
            tensor = tensor + torch.randn_like(tensor) * 0.01
            items.append(tensor)
        items.append(self.labels[idx])
        return tuple(items)


def _train_run(stream_monitoring: bool, seed: int = 42) -> dict:
    """
    Train a small LINet for a few epochs and return the history.

    Uses deterministic seeding so two runs with the same flag produce
    identical results, and the test can compare across flags.
    """
    set_seed(seed, deterministic=True)

    # Build dataset AFTER seeding so data is identical across runs
    dataset = AugmentedTupleDataset(NUM_TRAIN, num_streams=2)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = LINet(
        block=LIBasicBlock,
        layers=[1, 1, 1, 1],
        stream_in_channels=[3, 1],
        num_classes=NUM_CLASSES,
        device='cpu',
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.compile(optimizer=optimizer, loss='cross_entropy')

    history = model.fit(
        train_loader=train_loader,
        val_loader=None,
        epochs=EPOCHS,
        verbose=False,
        stream_monitoring=stream_monitoring,
        stream_eval_freq=1,
        stream_eval_samples=12,
    )
    return history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamMonitoringRNG:
    """Verify stream_monitoring=True produces identical training to False."""

    def test_training_loss_identical(self):
        """Per-epoch training loss must be bitwise identical with/without monitoring."""
        hist_off = _train_run(stream_monitoring=False)
        hist_on = _train_run(stream_monitoring=True)

        for epoch in range(EPOCHS):
            loss_off = hist_off['train_loss'][epoch]
            loss_on = hist_on['train_loss'][epoch]
            assert loss_off == loss_on, (
                f"Epoch {epoch + 1}: train_loss differs — "
                f"monitoring=False: {loss_off}, monitoring=True: {loss_on}"
            )

    def test_training_accuracy_identical(self):
        """Per-epoch training accuracy must be identical with/without monitoring."""
        hist_off = _train_run(stream_monitoring=False)
        hist_on = _train_run(stream_monitoring=True)

        for epoch in range(EPOCHS):
            acc_off = hist_off['train_accuracy'][epoch]
            acc_on = hist_on['train_accuracy'][epoch]
            assert acc_off == acc_on, (
                f"Epoch {epoch + 1}: train_accuracy differs — "
                f"monitoring=False: {acc_off}, monitoring=True: {acc_on}"
            )

    def test_rng_state_preserved(self):
        """
        Directly verify that _evaluate_stream_contributions restores RNG state.
        """
        set_seed(42, deterministic=True)

        dataset = AugmentedTupleDataset(NUM_TRAIN, num_streams=2)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = LINet(
            block=LIBasicBlock,
            layers=[1, 1, 1, 1],
            stream_in_channels=[3, 1],
            num_classes=NUM_CLASSES,
            device='cpu',
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.compile(optimizer=optimizer, loss='cross_entropy')

        # Capture RNG state before monitoring
        np_state_before = np.random.get_state()[1].copy()
        torch_state_before = torch.random.get_rng_state().clone()

        # Run monitoring evaluation
        model._evaluate_stream_contributions(loader)

        # Capture RNG state after monitoring
        np_state_after = np.random.get_state()[1].copy()
        torch_state_after = torch.random.get_rng_state().clone()

        assert np.array_equal(np_state_before, np_state_after), (
            "numpy RNG state was not restored after _evaluate_stream_contributions"
        )
        assert torch.equal(torch_state_before, torch_state_after), (
            "torch RNG state was not restored after _evaluate_stream_contributions"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
