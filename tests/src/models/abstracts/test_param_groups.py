"""Tests for get_stream_parameter_groups() parameter routing."""

import pytest
import torch

from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock


@pytest.fixture
def model():
    return LINet(
        block=LIBasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=10,
        stream_input_channels=[3, 1],
        device='cpu'
    )


class TestClassifierRouting:
    """Verify fc/fc_streams land in the integration group, not the BN group."""

    def test_fc_in_integration_group_split_mode(self, model):
        """When integration_weight_decay is set, fc should be in integration group."""
        groups = model.get_stream_parameter_groups(
            stream_lrs=[2e-4, 7e-4],
            shared_lr=5e-4,
            shared_weight_decay=0.0,
            integration_weight_decay=1e-3
        )
        # 4 groups: stream0, stream1, integration, other(BN)
        assert len(groups) == 4

        integration_group = groups[2]
        other_group = groups[3]

        # Collect param ids
        integration_ids = {id(p) for p in integration_group['params']}
        other_ids = {id(p) for p in other_group['params']}

        # fc params should be in integration group
        for name, param in model.named_parameters():
            if name.startswith('fc.'):
                assert id(param) in integration_ids, (
                    f"{name} should be in integration group, not other"
                )
                assert id(param) not in other_ids

    def test_fc_gets_integration_wd(self, model):
        """fc params should inherit integration_weight_decay, not 0."""
        groups = model.get_stream_parameter_groups(
            stream_lrs=[2e-4, 7e-4],
            shared_lr=5e-4,
            shared_weight_decay=0.0,
            integration_weight_decay=1e-3
        )
        integration_group = groups[2]
        assert integration_group['weight_decay'] == 1e-3

        # Verify fc.weight is in this group
        fc_weight = dict(model.named_parameters())['fc.weight']
        assert any(p is fc_weight for p in integration_group['params'])

    def test_bn_params_not_in_integration_group(self, model):
        """BN gamma/beta (1D params) should stay in other group with WD=0."""
        groups = model.get_stream_parameter_groups(
            stream_lrs=[2e-4, 7e-4],
            shared_lr=5e-4,
            shared_weight_decay=0.0,
            integration_weight_decay=1e-3
        )
        other_group = groups[3]
        assert other_group['weight_decay'] == 0.0

        # All params in other group should be 1D (BN scale/shift)
        for param in other_group['params']:
            assert param.dim() == 1, (
                f"Non-BN param (dim={param.dim()}) found in other group"
            )

    def test_all_params_accounted_for(self, model):
        """Every model parameter should appear in exactly one group."""
        groups = model.get_stream_parameter_groups(
            stream_lrs=[2e-4, 7e-4],
            shared_lr=5e-4,
            integration_weight_decay=1e-3
        )
        group_ids = []
        for g in groups:
            group_ids.extend(id(p) for p in g['params'])

        model_ids = [id(p) for p in model.parameters()]
        assert sorted(group_ids) == sorted(model_ids), "Parameter count mismatch"

    def test_legacy_mode_unchanged(self, model):
        """Without integration_weight_decay, fc merges with integration+BN (3 groups)."""
        groups = model.get_stream_parameter_groups(
            stream_lrs=[2e-4, 7e-4],
            shared_lr=5e-4,
        )
        # Legacy: stream0, stream1, shared (all non-stream merged)
        assert len(groups) == 3

        # All params still accounted for
        group_count = sum(len(g['params']) for g in groups)
        model_count = len(list(model.parameters()))
        assert group_count == model_count
