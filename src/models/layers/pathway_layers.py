"""Basic pathway layers for multi-stream neural networks."""

import torch.nn as nn


class PathwayBlock(nn.Module):
    """Basic building block for individual pathways."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ColorPathway(nn.Module):
    """Color (parvocellular-inspired) pathway for detailed color processing."""
    
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            PathwayBlock(input_channels, hidden_dim),
            PathwayBlock(hidden_dim, hidden_dim),
            PathwayBlock(hidden_dim, hidden_dim * 2)
        )
    
    def forward(self, x):
        return self.layers(x)


class BrightnessPathway(nn.Module):
    """Brightness (magnocellular-inspired) pathway for motion and brightness."""
    
    def __init__(self, input_channels=1, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            PathwayBlock(input_channels, hidden_dim),
            PathwayBlock(hidden_dim, hidden_dim),
            PathwayBlock(hidden_dim, hidden_dim * 2)
        )
    
    def forward(self, x):
        return self.layers(x)
