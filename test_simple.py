#!/usr/bin/env python3
"""
Quick test demonstrating the unified progress bar feature.
Shows T_loss, T_acc, V_loss, V_acc in a single progress bar per epoch.
"""
import sys
import numpy as np
from src.models.basic_multi_channel.base_multi_channel_network import BaseMultiChannelNetwork

sys.path.append('.')

print("🧪 Quick unified progress bar demonstration...")

# Create dummy data
n_samples, input_size = 200, 32
color_data = np.random.randn(n_samples, input_size).astype(np.float32)
brightness_data = np.random.randn(n_samples, input_size).astype(np.float32)
labels = np.random.randint(0, 5, n_samples)

# Split into train/validation
split_idx = int(0.8 * n_samples)
train_color, val_color = color_data[:split_idx], color_data[split_idx:]
train_brightness, val_brightness = brightness_data[:split_idx], brightness_data[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

print(f"📊 Data: {len(train_color)} train, {len(val_color)} validation samples")

# Create and train model
model = BaseMultiChannelNetwork(
    color_input_size=input_size, 
    brightness_input_size=input_size, 
    hidden_sizes=[64, 32], 
    num_classes=5
)

print("� Training with unified progress bar (T_loss, T_acc, V_loss, V_acc):")
try:
    model.fit(
        train_color, train_brightness, train_labels,
        val_color_data=val_color, 
        val_brightness_data=val_brightness, 
        val_labels=val_labels,
        epochs=2,
        batch_size=16,
        verbose=1
    )
    print("✅ Unified progress bar test successful!")
    print("✨ Feature shows all metrics in ONE bar instead of separate train/val bars")
except Exception as e:
    print(f"❌ Error: {e}")
