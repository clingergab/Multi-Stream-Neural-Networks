"""
Pluggable classifier heads for LINet3.

Each head is an ``nn.Module`` that maps penultimate features ``(B, feat_dim)``
to logits ``(B, num_classes)``. ``LINet`` selects the head at construction via
the ``classifier_head`` kwarg and assigns it to ``self.fc`` so the backbone
forward stays as ``logits = self.fc(features)`` regardless of head type.

Adding a new head: define an ``nn.Module`` here, export it from this package's
``__init__.py``, and add an ``elif`` branch in ``LINet._build_network``.
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Standard linear classifier: ``nn.Linear(feat_dim, num_classes)``."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MaxoutHead(nn.Module):
    """K-sub-node max-out classifier.

    Each class has K parallel sub-classifiers; the output logit for class c is
    the max over its K sub-node activations. Trained end-to-end with standard
    cross-entropy on the post-max logits. With K=1 this reduces mathematically
    to a standard linear classifier (max of a singleton is the singleton).
    """

    def __init__(self, feat_dim: int, num_classes: int, num_subnodes: int = 3):
        super().__init__()
        assert num_subnodes >= 1, f"num_subnodes must be >= 1, got {num_subnodes}"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_subnodes = num_subnodes
        self.fc = nn.Linear(feat_dim, num_classes * num_subnodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, C*K)
        return logits.view(-1, self.num_classes, self.num_subnodes).amax(dim=2)  # (B, C)
