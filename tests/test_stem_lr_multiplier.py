"""
Tests for stem_lr_multiplier in get_stream_parameter_groups and create_stream_optimizer.

Verifies:
1. Backward compatibility: stem_lr_multiplier=1.0 returns the same 4-group structure
2. 6-group split: stem_lr_multiplier!=1.0 creates separate stem groups with scaled LR/WD
3. Gradient flow: optimizer with stem multiplier correctly updates stem params
"""

import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.linear_integration.li_net3 import li_resnet18
from src.training.optimizers import create_stream_optimizer


DEVICE = 'cpu'
NUM_CLASSES = 10
STREAM_LRS = [1e-4, 2e-4]
STREAM_WDS = [1e-4, 2e-4]
SHARED_LR = 1.5e-4
INTEGRATION_WD = 1e-4


def test_backward_compatibility_default():
    """stem_lr_multiplier=1.0 returns the existing 4-group structure."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)

    groups = model.get_stream_parameter_groups(
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
        stem_lr_multiplier=1.0,
    )

    assert len(groups) == 4, f"Expected 4 groups with multiplier=1.0, got {len(groups)}"

    # conv1 params should be in the stream groups (groups 0 and 1), not separate
    conv1_param_names = {
        id(p) for name, p in model.named_parameters() if name.startswith('conv1.stream_')
    }
    group_0_1_ids = set()
    for g in groups[:2]:
        for p in g['params']:
            group_0_1_ids.add(id(p))

    assert conv1_param_names.issubset(group_0_1_ids), "conv1 params should be in stream groups at multiplier=1.0"

    # LRs should be standard stream LRs
    assert groups[0]['lr'] == STREAM_LRS[0]
    assert groups[1]['lr'] == STREAM_LRS[1]


def test_backward_compatibility_optimizer():
    """create_stream_optimizer works with default stem_lr_multiplier."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)

    optimizer = create_stream_optimizer(
        model,
        optimizer_type='adamw',
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
    )

    assert len(optimizer.param_groups) == 4


def test_six_group_split():
    """stem_lr_multiplier=10.0 creates 6 groups with correct LR and WD."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    multiplier = 10.0

    groups = model.get_stream_parameter_groups(
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
        stem_lr_multiplier=multiplier,
    )

    assert len(groups) == 6, f"Expected 6 groups with multiplier=10.0, got {len(groups)}"

    # Groups 0-1: stem (conv1) with scaled LR and inverse-scaled WD
    assert abs(groups[0]['lr'] - STREAM_LRS[0] * multiplier) < 1e-10, \
        f"Stem group 0 LR: expected {STREAM_LRS[0] * multiplier}, got {groups[0]['lr']}"
    assert abs(groups[1]['lr'] - STREAM_LRS[1] * multiplier) < 1e-10, \
        f"Stem group 1 LR: expected {STREAM_LRS[1] * multiplier}, got {groups[1]['lr']}"
    assert abs(groups[0]['weight_decay'] - STREAM_WDS[0] / multiplier) < 1e-10, \
        f"Stem group 0 WD: expected {STREAM_WDS[0] / multiplier}, got {groups[0]['weight_decay']}"
    assert abs(groups[1]['weight_decay'] - STREAM_WDS[1] / multiplier) < 1e-10, \
        f"Stem group 1 WD: expected {STREAM_WDS[1] / multiplier}, got {groups[1]['weight_decay']}"

    # Groups 2-3: backbone with original LR and WD
    assert groups[2]['lr'] == STREAM_LRS[0]
    assert groups[3]['lr'] == STREAM_LRS[1]
    assert groups[2]['weight_decay'] == STREAM_WDS[0]
    assert groups[3]['weight_decay'] == STREAM_WDS[1]

    # Groups 4-5: integration and other (shared_lr)
    assert groups[4]['lr'] == SHARED_LR
    assert groups[5]['lr'] == SHARED_LR


def test_stem_groups_contain_only_conv1():
    """Stem groups should contain only conv1 stream params, not layer1-4."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)

    groups = model.get_stream_parameter_groups(
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
        stem_lr_multiplier=10.0,
    )

    # Build name lookup
    param_to_name = {id(p): name for name, p in model.named_parameters()}

    # Check stem groups (0, 1) contain only conv1 params
    for group_idx in [0, 1]:
        for p in groups[group_idx]['params']:
            name = param_to_name[id(p)]
            assert name.startswith('conv1.'), \
                f"Stem group {group_idx} contains non-conv1 param: {name}"

    # Check backbone groups (2, 3) contain no conv1 params
    for group_idx in [2, 3]:
        for p in groups[group_idx]['params']:
            name = param_to_name[id(p)]
            assert not name.startswith('conv1.'), \
                f"Backbone group {group_idx} contains conv1 param: {name}"


def test_all_params_accounted_for():
    """Every parameter appears in exactly one group."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)

    groups = model.get_stream_parameter_groups(
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
        stem_lr_multiplier=10.0,
    )

    all_model_params = set(id(p) for p in model.parameters())
    grouped_params = []
    for g in groups:
        for p in g['params']:
            grouped_params.append(id(p))

    grouped_set = set(grouped_params)
    assert len(grouped_params) == len(grouped_set), "Duplicate parameters across groups!"
    assert grouped_set == all_model_params, \
        f"Missing {len(all_model_params - grouped_set)} params, extra {len(grouped_set - all_model_params)} params"


def test_gradient_flow_with_multiplier():
    """Optimizer with stem_lr_multiplier creates correct groups and params get gradients."""
    model = li_resnet18(num_classes=NUM_CLASSES, stream_input_channels=[3, 1], device=DEVICE)
    model.train()

    optimizer = create_stream_optimizer(
        model,
        optimizer_type='adamw',
        stream_lrs=STREAM_LRS,
        stream_weight_decays=STREAM_WDS,
        shared_lr=SHARED_LR,
        integration_weight_decay=INTEGRATION_WD,
        stem_lr_multiplier=10.0,
    )

    assert len(optimizer.param_groups) == 6
    assert abs(optimizer.param_groups[0]['lr'] - STREAM_LRS[0] * 10.0) < 1e-10

    # Forward + backward
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    targets = torch.randint(0, NUM_CLASSES, (2,))

    optimizer.zero_grad()
    logits = model([rgb, depth])
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    # conv1 stream weights should have gradients
    assert model.conv1.stream_weights[0].grad is not None
    assert model.conv1.stream_weights[0].grad.abs().sum() > 0


ALL_TESTS = [
    test_backward_compatibility_default,
    test_backward_compatibility_optimizer,
    test_six_group_split,
    test_stem_groups_contain_only_conv1,
    test_all_params_accounted_for,
    test_gradient_flow_with_multiplier,
]


if __name__ == "__main__":
    print("=" * 70)
    print("TEST SUITE: stem_lr_multiplier")
    print("=" * 70)

    passed = failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failed += 1

    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
