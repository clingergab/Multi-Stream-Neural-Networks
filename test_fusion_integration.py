"""
Test fusion integration in MCResNet.

Verifies that:
1. MCResNet can be created with different fusion types
2. Forward pass works correctly with all fusion types
3. Output dimensions are correct
4. Fusion modules are properly initialized
"""

import torch
from src.models.multi_channel import mc_resnet18

def test_fusion_integration():
    """Test that fusion integration works correctly."""

    print("Testing fusion integration in MCResNet...")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    num_classes = 10
    img_size = 224
    stream1_channels = 3  # RGB
    stream2_channels = 1  # Depth

    # Create sample data
    stream1_input = torch.randn(batch_size, stream1_channels, img_size, img_size)
    stream2_input = torch.randn(batch_size, stream2_channels, img_size, img_size)

    fusion_types = ['concat', 'weighted', 'gated']

    for fusion_type in fusion_types:
        print(f"\n{fusion_type.upper()} Fusion:")
        print("-" * 60)

        # Create model with specific fusion type
        model = mc_resnet18(
            num_classes=num_classes,
            stream1_input_channels=stream1_channels,
            stream2_input_channels=stream2_channels,
            fusion_type=fusion_type,
            device='cpu'
        )

        # Verify fusion module exists and has correct type
        assert hasattr(model, 'fusion'), f"Model missing fusion module for {fusion_type}"
        print(f"✓ Fusion module created: {type(model.fusion).__name__}")

        # Verify fusion type property
        assert model.fusion_strategy == fusion_type, f"Fusion type mismatch: {model.fusion_strategy} != {fusion_type}"
        print(f"✓ Fusion type: {model.fusion_strategy}")

        # Verify output dimension
        expected_feature_dim = 512  # ResNet18 final layer
        expected_output_dim = expected_feature_dim * 2  # All fusion types concatenate
        assert model.fusion.output_dim == expected_output_dim, \
            f"Output dim mismatch: {model.fusion.output_dim} != {expected_output_dim}"
        print(f"✓ Fusion output dim: {model.fusion.output_dim}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(stream1_input, stream2_input)

        # Verify output shape
        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
        print(f"✓ Forward pass successful: {output.shape}")

        # Verify output is valid (no NaN or Inf)
        assert not torch.isnan(output).any(), f"NaN values in output for {fusion_type}"
        assert not torch.isinf(output).any(), f"Inf values in output for {fusion_type}"
        print(f"✓ Output is valid (no NaN/Inf)")

        # For weighted and gated fusion, check learnable parameters
        if fusion_type == 'weighted':
            assert hasattr(model.fusion, 'stream1_weight'), "WeightedFusion missing stream1_weight"
            assert hasattr(model.fusion, 'stream2_weight'), "WeightedFusion missing stream2_weight"
            print(f"✓ Learnable weights initialized: stream1={model.fusion.stream1_weight.item():.4f}, "
                  f"stream2={model.fusion.stream2_weight.item():.4f}")
        elif fusion_type == 'gated':
            assert hasattr(model.fusion, 'gate_network'), "GatedFusion missing gate_network"
            print(f"✓ Gate network created with {sum(p.numel() for p in model.fusion.gate_network.parameters())} parameters")

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"✓ Total params: {total_params:,} (fusion: {fusion_params:,})")

    print("\n" + "=" * 60)
    print("✅ All fusion integration tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_fusion_integration()
