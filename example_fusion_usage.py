"""
Example: Using Different Fusion Strategies with MCResNet

This script demonstrates how to create and use MCResNet with different fusion strategies.
"""

import torch
from src.models.multi_channel import mc_resnet18

def main():
    print("=" * 70)
    print("MCResNet Fusion Strategies Demo")
    print("=" * 70)

    # Configuration
    num_classes = 27  # NYU Depth V2 scene classes
    batch_size = 8
    img_size = 224

    # Create sample data (RGB + Depth)
    rgb_data = torch.randn(batch_size, 3, img_size, img_size)
    depth_data = torch.randn(batch_size, 1, img_size, img_size)

    # Demonstrate each fusion type
    fusion_types = ['concat', 'weighted', 'gated']

    for fusion_type in fusion_types:
        print(f"\n{'─' * 70}")
        print(f"Fusion Type: {fusion_type.upper()}")
        print(f"{'─' * 70}")

        # Create model with specific fusion strategy
        model = mc_resnet18(
            num_classes=num_classes,
            stream1_input_channels=3,  # RGB
            stream2_input_channels=1,  # Depth
            fusion_type=fusion_type,
            dropout_p=0.3,
            device='cpu'
        )

        # Print fusion info
        print(f"Fusion module: {type(model.fusion).__name__}")
        print(f"Fusion output dim: {model.fusion.output_dim}")
        print(f"Fusion strategy: {model.fusion_strategy}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        fusion_params = sum(p.numel() for p in model.fusion.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Fusion parameters: {fusion_params:,}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(rgb_data, depth_data)
        print(f"Output shape: {output.shape}")

        # Show fusion-specific details
        if fusion_type == 'weighted':
            w1 = model.fusion.stream1_weight.item()
            w2 = model.fusion.stream2_weight.item()
            print(f"Stream1 weight: {w1:.4f}")
            print(f"Stream2 weight: {w2:.4f}")
        elif fusion_type == 'gated':
            gate_params = sum(p.numel() for p in model.fusion.gate_network.parameters())
            print(f"Gate network parameters: {gate_params:,}")

    print(f"\n{'=' * 70}")
    print("Training Example - Standard Optimization")
    print(f"{'=' * 70}\n")

    # Show how to use in training
    print("# Create model with gated fusion")
    print("model = mc_resnet18(")
    print("    num_classes=27,")
    print("    stream1_input_channels=3,")
    print("    stream2_input_channels=1,")
    print("    fusion_type='gated',  # ← Change this to try different strategies")
    print("    dropout_p=0.3")
    print(")")
    print()
    print("# Compile with standard optimization")
    print("model.compile(")
    print("    optimizer='adamw',")
    print("    learning_rate=1e-4,")
    print("    weight_decay=2e-2")
    print(")")
    print()
    print("# Train")
    print("history = model.fit(")
    print("    train_loader=train_loader,")
    print("    val_loader=val_loader,")
    print("    epochs=50")
    print(")")

    print(f"\n{'=' * 70}")
    print("Training Example - Stream-Specific Optimization ✨")
    print(f"{'=' * 70}\n")
    print("# Address pathway imbalance with stream-specific learning rates")
    print("model = mc_resnet18(")
    print("    num_classes=27,")
    print("    stream1_input_channels=3,")
    print("    stream2_input_channels=1,")
    print("    fusion_type='weighted'")
    print(")")
    print()
    print("# Compile with stream-specific optimization")
    print("model.compile(")
    print("    optimizer='adamw',")
    print("    learning_rate=1e-4,         # Base LR for shared params")
    print("    weight_decay=2e-2,          # Base weight decay")
    print("    # Boost RGB pathway (weaker)")
    print("    stream1_lr=5e-4,            # 5x higher LR for RGB")
    print("    stream1_weight_decay=1e-3,  # Lighter regularization")
    print("    # Regularize depth pathway (stronger)")
    print("    stream2_lr=5e-5,            # 2x lower LR for depth")
    print("    stream2_weight_decay=5e-2   # Heavier regularization")
    print(")")
    print()
    print("# Train - pathways now balanced!")
    print("history = model.fit(")
    print("    train_loader=train_loader,")
    print("    val_loader=val_loader,")
    print("    epochs=100")
    print(")")
    print()
    print("# Analyze pathway balance")
    print("analysis = model.analyze_pathways(val_loader)")
    print("print(f\"RGB: {analysis['accuracy']['color_contribution']:.1%}\")")
    print("print(f\"Depth: {analysis['accuracy']['brightness_contribution']:.1%}\")")
    print()

    print(f"\n{'=' * 70}")
    print("Summary: Multi-Stream Improvements Available")
    print(f"{'=' * 70}\n")
    print("✅ Fusion Strategies:")
    print("   • concat:   Simple concatenation (baseline)")
    print("   • weighted: Learned scalar weights per stream")
    print("   • gated:    Adaptive per-sample gating via MLP")
    print()
    print("✅ Stream-Specific Optimization:")
    print("   • stream1_lr, stream2_lr: Different learning rates")
    print("   • stream1_weight_decay, stream2_weight_decay: Different regularization")
    print("   • Automatic parameter grouping by stream")
    print()
    print("✅ Use Both Together:")
    print("   • WeightedFusion + stream-specific LR = Best results")
    print("   • GatedFusion + stream-specific WD = Adaptive + balanced")
    print()

if __name__ == "__main__":
    main()
