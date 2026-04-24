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
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Standard linear classifier: ``nn.Linear(feat_dim, num_classes)``."""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def diversity_loss(self, weight: float = 0.0) -> torch.Tensor:
        """Always zero for LinearHead (one hyperplane per class, nothing to diversify).

        Defined so the training loop can unconditionally call
        ``model.fc.diversity_loss(weight=...)`` regardless of head type.
        """
        return torch.zeros((), device=self.fc.weight.device)


class MaxoutHead(nn.Module):
    """K-sub-node max-out classifier with anti-dead-unit counter-measures.

    Each class has K parallel sub-classifiers; the output logit for class c is
    the max over its K sub-node activations. Trained end-to-end with standard
    cross-entropy on the post-max logits. With K=1 this reduces mathematically
    to a standard linear classifier.

    Gradient starvation and the two counter-measures
    -----------------------------------------------
    Naive max-out suffers from winner-takes-all: only the argmax sub-node
    receives gradient per (sample, class), so losing sub-nodes stay frozen at
    init forever. Cell 19.12 on SUN RGB-D confirmed this empirically — K=2
    gave mean normalized sub-node entropy = 0.000 on hpo_train with all 19
    classes having dead (<5%) sub-nodes. Two counter-measures are applied:

    1. **Asymmetric init** — a Gaussian perturbation added post-construction
       with std ~3x the default Kaiming std. This alone is insufficient
       (winner-takes-all is a *dynamic*, not an init problem), but it gives
       sub-nodes a larger initial spread which reduces the probability that
       one sub-node wins on every sample early.

    2. **Sub-node dropout during training** — with probability ``subnode_dropout``
       per (batch, class, sub-node), mask the sub-node to ``-inf`` before
       argmax. Forces runner-up sub-nodes to occasionally win and receive
       gradient. Disabled at eval time; at K=1 it is unconditionally a no-op.

    Both counter-measures are skipped when ``num_subnodes == 1`` so that
    MaxoutHead(K=1) stays byte-identical to LinearHead under matched seeding
    (preserves the construction-equivalence test).

    A third, opt-in counter-measure — ``diversity_loss(weight)`` — is exposed
    as a method for the training loop to add to the CE loss before each
    ``.backward()``. It penalizes off-diagonal |cos-sim| between sub-node
    weight vectors per class, giving gradient to ALL sub-nodes (not just the
    argmax winner). See the method docstring.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_subnodes: int = 3,
        subnode_dropout: float = 0.1,
        init_perturb_std: float = 0.1,
    ):
        super().__init__()
        assert num_subnodes >= 1, f"num_subnodes must be >= 1, got {num_subnodes}"
        assert 0.0 <= subnode_dropout < 1.0, (
            f"subnode_dropout must be in [0, 1), got {subnode_dropout}"
        )
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_subnodes = num_subnodes
        self.subnode_dropout = subnode_dropout
        self.init_perturb_std = init_perturb_std
        self.fc = nn.Linear(feat_dim, num_classes * num_subnodes)
        if num_subnodes > 1 and init_perturb_std > 0.0:
            with torch.no_grad():
                self.fc.weight.data.add_(init_perturb_std * torch.randn_like(self.fc.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, C*K)
        sub = logits.view(-1, self.num_classes, self.num_subnodes)  # (B, C, K)
        if self.training and self.num_subnodes > 1 and self.subnode_dropout > 0.0:
            drop = torch.rand_like(sub) < self.subnode_dropout  # (B, C, K) True = mask
            all_dropped = drop.all(dim=2, keepdim=True)          # (B, C, 1)
            drop = drop & ~all_dropped                            # keep at least one
            sub = sub.masked_fill(drop, float('-inf'))
        return sub.amax(dim=2)  # (B, C)

    def diversity_loss(self, weight: float = 0.0) -> torch.Tensor:
        """Per-class cosine-diversity penalty on sub-node weight vectors.

        For each class, computes the mean absolute off-diagonal cosine similarity
        between its K sub-node weight vectors, then averages over classes. Scaled
        by ``weight`` and returned as a scalar tensor to be added to the CE loss
        before ``.backward()``. Minimizing pushes sub-node weights into distinct
        directions in feature space, complementing forward-time sub-node dropout
        against winner-takes-all gradient starvation.

        Returns a zero scalar when ``num_subnodes == 1`` or ``weight == 0.0``.
        """
        if self.num_subnodes == 1 or weight == 0.0:
            return torch.zeros((), device=self.fc.weight.device)
        W = self.fc.weight.view(self.num_classes, self.num_subnodes, self.feat_dim)
        W_norm = F.normalize(W, dim=-1)                                    # unit vectors
        sim = torch.matmul(W_norm, W_norm.transpose(-1, -2))               # (C, K, K)
        K = self.num_subnodes
        off_diag = 1.0 - torch.eye(K, device=sim.device, dtype=sim.dtype)  # (K, K)
        penalty_per_class = (sim.abs() * off_diag).sum(dim=(1, 2)) / (K * (K - 1))
        return penalty_per_class.mean() * weight
