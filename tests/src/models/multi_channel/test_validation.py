"""
Validation tests for Multi-Channel ResNet implementation.

These tests verify:
1. Parameter count matches expected values
2. Output shapes are correct
3. Pathways remain independent
4. Gradients flow to both pathways
"""

import pytest
import torch
import torch.nn as nn
from src.models.multi_channel.mc_resnet import MCResNet, mc_resnet18, mc_resnet34, mc_resnet50
from src.models.multi_channel.blocks import MCBasicBlock, MCBottleneck
from src.models.utils.gradient_monitor import GradientMonitor


class TestParameterCount:
    """Test that parameter counts match expected values."""

    def test_mcresnet18_parameter_count(self):
        """MCResNet18 should have ~23.4M parameters (2x standard ResNet18)."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: ~23.4M parameters
        # Standard ResNet18: ~11.7M
        # MCResNet18: 2 × 11.7M ≈ 23.4M
        expected_params = 23_400_000
        tolerance = 100_000  # Allow 100k tolerance

        assert abs(total_params - expected_params) < tolerance, \
            f"Expected ~{expected_params:,} params, got {total_params:,}"

    def test_mcresnet34_parameter_count(self):
        """MCResNet34 should have ~44M parameters (2x standard ResNet34)."""
        model = mc_resnet34(num_classes=1000, stream1_channels=3, stream2_channels=1)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: ~44M parameters
        expected_params = 44_000_000
        tolerance = 200_000

        assert abs(total_params - expected_params) < tolerance, \
            f"Expected ~{expected_params:,} params, got {total_params:,}"

    def test_mcresnet50_parameter_count(self):
        """MCResNet50 should have ~50M parameters (2x standard ResNet50)."""
        model = mc_resnet50(num_classes=1000, stream1_channels=3, stream2_channels=1)
        total_params = sum(p.numel() for p in model.parameters())

        # Expected: ~50M parameters
        expected_params = 50_000_000
        tolerance = 500_000

        assert abs(total_params - expected_params) < tolerance, \
            f"Expected ~{expected_params:,} params, got {total_params:,}"

    def test_dual_pathway_parameters(self):
        """Verify that both pathways have parameters."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)

        stream1_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'stream1' in name or 'color' in name
        )
        stream2_params = sum(
            p.numel() for name, p in model.named_parameters()
            if 'stream2' in name or 'brightness' in name
        )

        # Both pathways should have substantial parameters
        assert stream1_params > 1_000_000, "Stream1 pathway has too few parameters"
        assert stream2_params > 1_000_000, "Stream2 pathway has too few parameters"


class TestOutputShapes:
    """Test that output shapes are correct."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("num_classes", [10, 100, 1000])
    def test_output_shape(self, batch_size, num_classes):
        """Test output shape for various batch sizes and class counts."""
        model = mc_resnet18(num_classes=num_classes, stream1_channels=3, stream2_channels=1)
        model.eval()

        stream1_input = torch.randn(batch_size, 3, 224, 224)
        stream2_input = torch.randn(batch_size, 1, 224, 224)

        with torch.no_grad():
            output = model(stream1_input, stream2_input)

        assert output.shape == (batch_size, num_classes), \
            f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"

    @pytest.mark.parametrize("input_size", [(224, 224), (112, 112), (56, 56)])
    def test_variable_input_sizes(self, input_size):
        """Test that model handles variable input sizes."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)
        model.eval()

        h, w = input_size
        stream1_input = torch.randn(2, 3, h, w)
        stream2_input = torch.randn(2, 1, h, w)

        with torch.no_grad():
            output = model(stream1_input, stream2_input)

        assert output.shape == (2, 1000), \
            f"Expected shape (2, 1000), got {output.shape}"

    def test_intermediate_feature_shapes(self):
        """Test intermediate feature map shapes through the network."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)
        model.eval()

        stream1_input = torch.randn(1, 3, 224, 224)
        stream2_input = torch.randn(1, 1, 224, 224)

        # Track shapes through network
        with torch.no_grad():
            # Initial conv
            s1, s2 = model.conv1(stream1_input, stream2_input)
            assert s1.shape[2:] == (112, 112), f"After conv1: {s1.shape}"

            s1, s2 = model.bn1(s1, s2)
            s1, s2 = model.relu(s1, s2)
            s1, s2 = model.maxpool(s1, s2)
            assert s1.shape[2:] == (56, 56), f"After maxpool: {s1.shape}"

            # Layer1 (no spatial reduction)
            s1, s2 = model.layer1(s1, s2)
            assert s1.shape[2:] == (56, 56), f"After layer1: {s1.shape}"

            # Layer2 (spatial reduction)
            s1, s2 = model.layer2(s1, s2)
            assert s1.shape[2:] == (28, 28), f"After layer2: {s1.shape}"

            # Layer3 (spatial reduction)
            s1, s2 = model.layer3(s1, s2)
            assert s1.shape[2:] == (14, 14), f"After layer3: {s1.shape}"

            # Layer4 (spatial reduction)
            s1, s2 = model.layer4(s1, s2)
            assert s1.shape[2:] == (7, 7), f"After layer4: {s1.shape}"

            # Avgpool
            s1, s2 = model.avgpool(s1, s2)
            assert s1.shape[2:] == (1, 1), f"After avgpool: {s1.shape}"


class TestPathwayIndependence:
    """Test that pathways remain independent during forward pass."""

    def test_streams_dont_mix(self):
        """Verify that stream1 and stream2 don't cross-contaminate."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)
        model.eval()

        # Create distinct inputs
        stream1_input = torch.ones(1, 3, 224, 224)
        stream2_input = torch.zeros(1, 1, 224, 224)

        with torch.no_grad():
            output1 = model(stream1_input, stream2_input)

        # Swap inputs
        stream1_input_swapped = torch.zeros(1, 3, 224, 224)
        stream2_input_swapped = torch.ones(1, 1, 224, 224)

        with torch.no_grad():
            output2 = model(stream1_input_swapped, stream2_input_swapped)

        # Outputs should be different (pathways are independent)
        assert not torch.allclose(output1, output2, atol=1e-6), \
            "Outputs should differ when inputs are swapped (pathways should be independent)"

    def test_single_pathway_forward(self):
        """Test that single-pathway forward methods work correctly."""
        model = mc_resnet18(num_classes=1000, stream1_channels=3, stream2_channels=1)
        model.eval()

        stream1_input = torch.randn(2, 3, 224, 224)

        # Test stream1-only forward through a block
        with torch.no_grad():
            # Test conv layer
            stream1_out = model.conv1.forward_stream1(stream1_input)
            assert stream1_out.shape == (2, 64, 112, 112)

            # Test that it produces same result as full forward
            s1_full, _ = model.conv1(stream1_input, torch.randn(2, 1, 224, 224))
            assert torch.allclose(stream1_out, s1_full, atol=1e-6)


class TestGradientFlow:
    """Test that gradients flow to both pathways."""

    def test_both_pathways_receive_gradients(self):
        """Verify that both pathways receive gradients during backprop."""
        model = mc_resnet18(num_classes=10, stream1_channels=3, stream2_channels=1)
        model.train()

        stream1_input = torch.randn(4, 3, 224, 224, requires_grad=True)
        stream2_input = torch.randn(4, 1, 224, 224, requires_grad=True)
        targets = torch.randint(0, 10, (4,))

        # Forward pass
        outputs = model(stream1_input, stream2_input)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        # Backward pass
        loss.backward()

        # Check that both pathways have gradients
        stream1_has_grad = False
        stream2_has_grad = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'stream1' in name or 'color' in name:
                    stream1_has_grad = True
                if 'stream2' in name or 'brightness' in name:
                    stream2_has_grad = True

        assert stream1_has_grad, "Stream1 pathway should receive gradients"
        assert stream2_has_grad, "Stream2 pathway should receive gradients"

    def test_gradient_magnitudes_reasonable(self):
        """Test that gradient magnitudes are in reasonable range."""
        model = mc_resnet18(num_classes=10, stream1_channels=3, stream2_channels=1)
        model.train()

        stream1_input = torch.randn(4, 3, 224, 224)
        stream2_input = torch.randn(4, 1, 224, 224)
        targets = torch.randint(0, 10, (4,))

        # Forward + backward
        outputs = model(stream1_input, stream2_input)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()

        # Use gradient monitor
        monitor = GradientMonitor(model, stream1_name="stream1", stream2_name="stream2")
        stats = monitor.compute_pathway_stats()

        # Gradients should exist and be non-zero
        assert stats['stream1_grad_norm'] > 0, "Stream1 gradients should be non-zero"
        assert stats['stream2_grad_norm'] > 0, "Stream2 gradients should be non-zero"

        # Ratio should be reasonable (not extreme imbalance)
        assert 0.1 < stats['ratio'] < 10.0, \
            f"Gradient ratio ({stats['ratio']:.2f}) indicates potential pathway collapse"

    def test_gradient_monitor_zero_overhead(self):
        """Verify gradient monitor has zero overhead when not used."""
        import time

        model = mc_resnet18(num_classes=10, stream1_channels=3, stream2_channels=1)
        model.train()

        stream1_input = torch.randn(4, 3, 224, 224)
        stream2_input = torch.randn(4, 1, 224, 224)
        targets = torch.randint(0, 10, (4,))

        # Time without monitoring
        start = time.time()
        for _ in range(10):
            outputs = model(stream1_input, stream2_input)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            model.zero_grad()
        time_without_monitor = time.time() - start

        # Create monitor but don't use it
        monitor = GradientMonitor(model)

        # Time with monitor created but not called
        start = time.time()
        for _ in range(10):
            outputs = model(stream1_input, stream2_input)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            model.zero_grad()
        time_with_inactive_monitor = time.time() - start

        # Should be nearly identical (< 5% difference)
        overhead = (time_with_inactive_monitor - time_without_monitor) / time_without_monitor
        assert overhead < 0.05, \
            f"Inactive gradient monitor should have zero overhead (got {overhead*100:.1f}%)"


class TestChannelFlexibility:
    """Test that model handles different channel configurations."""

    @pytest.mark.parametrize("stream1_channels,stream2_channels", [
        (3, 1),   # RGB + Grayscale
        (3, 3),   # RGB + RGB
        (1, 1),   # Grayscale + Grayscale
        (4, 1),   # RGBA + Grayscale
    ])
    def test_variable_channel_counts(self, stream1_channels, stream2_channels):
        """Test various channel configurations."""
        model = mc_resnet18(
            num_classes=10,
            stream1_channels=stream1_channels,
            stream2_channels=stream2_channels
        )
        model.eval()

        stream1_input = torch.randn(2, stream1_channels, 224, 224)
        stream2_input = torch.randn(2, stream2_channels, 224, 224)

        with torch.no_grad():
            output = model(stream1_input, stream2_input)

        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"


class TestResidualConnections:
    """Test that residual connections work correctly."""

    def test_identity_mapping_preserved(self):
        """Verify that identity mappings work when dimensions match."""
        model = mc_resnet18(num_classes=10, stream1_channels=3, stream2_channels=1)

        # Get first block from layer1 (no downsampling)
        block = model.layer1[0]

        # Create input with matching dimensions
        stream1_input = torch.randn(1, 64, 56, 56)
        stream2_input = torch.randn(1, 64, 56, 56)

        # Forward through block
        s1_out, s2_out = block(stream1_input, stream2_input)

        # Output should have same spatial dimensions
        assert s1_out.shape == stream1_input.shape
        assert s2_out.shape == stream2_input.shape

    def test_downsampling_works(self):
        """Verify that downsampling adjusts identity correctly."""
        model = mc_resnet18(num_classes=10, stream1_channels=3, stream2_channels=1)

        # Get first block from layer2 (has downsampling)
        block = model.layer2[0]

        # Input from previous layer
        stream1_input = torch.randn(1, 64, 56, 56)
        stream2_input = torch.randn(1, 64, 56, 56)

        # Forward through block with stride=2
        s1_out, s2_out = block(stream1_input, stream2_input)

        # Output should be spatially downsampled and have more channels
        assert s1_out.shape == (1, 128, 28, 28), f"Got {s1_out.shape}"
        assert s2_out.shape == (1, 128, 28, 28), f"Got {s2_out.shape}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
