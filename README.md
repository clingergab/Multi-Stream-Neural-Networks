# LINet: Linear Integration Networks for Multimodal Fusion

A biologically-inspired multi-stream neural network architecture that integrates complementary data modalities at the neuron level, applied to RGB-D scene classification.

## Overview

Most multimodal fusion approaches are ad-hoc: early fusion concatenates inputs before any processing, late fusion merges predictions after independent networks have already committed to separate representations. Neither mirrors how biological neurons actually combine information.

In the brain, a single neuron receives signals from multiple dendritic inputs and integrates them *before* applying a firing threshold -- not after. This integration-before-threshold mechanism is fundamental to how the nervous system combines complementary information streams.

![MSNN Neuron Integration](docs/images/msnn_neuron_integration.png)

**LINet applies this principle to deep learning.** Each neuron in the network maintains separate learned weight matrices for each input modality (e.g., RGB and Depth), plus a self-weight for the integrated pathway. These are combined via linear integration *inside* the neuron, before batch normalization and activation. The result is a network where fusion is not a single architectural decision point, but a continuous, learned process at every layer.

The current application is **scene classification on SUN RGB-D** (19-category standard benchmark), using RGB and Depth as complementary modalities from independent sensors.

## Architecture

![LINet Architecture](docs/images/linet_architecture.png)

LINet extends the ResNet backbone by replacing standard convolution layers with **LIConv2d** neurons. Each LIConv2d maintains:

- **Stream weights** (V_1, V_2, ...): One weight matrix per input modality
- **Self-weight** (V_prev): A weight matrix for the integrated pathway from the previous layer
- **Summation**: Linear combination of all stream outputs and the self-pathway, integrated before normalization

The full pipeline at each stage:

1. Each stream passes through its own spatial convolution (independent feature extraction)
2. Stream outputs are projected through learned linear weights (1x1 convolutions)
3. The integrated pathway's own projection is added
4. The sum passes through **LI-BatchNorm2d** and **LI-ReLU**
5. Individual streams continue forward independently to the next stage

This means every stream maintains its own representation throughout the network while continuously contributing to a shared integrated pathway. Final classification uses only the integrated pathway output.

## Key Features

- **Integration-before-threshold** -- Biologically motivated fusion at the neuron level, not a bolt-on fusion layer
- **N-stream flexibility** -- Works with any number of input modalities (2, 3, 4+)
- **GPU augmentation pipeline** -- Data augmentation runs on GPU with modality-specific transforms (color jitter for RGB, noise for Depth)
- **Modality dropout** -- Randomly drops entire modalities during training to build robustness
- **Pathway analysis** -- Tools to evaluate per-stream contributions, mixing weight evolution, and ablation studies
- **Channels-last optimization** -- NHWC memory format eliminates reformatting overhead for cuDNN

## Quick Start

```bash
git clone https://github.com/your-username/Multi-Stream-Neural-Networks.git
cd Multi-Stream-Neural-Networks
pip install -e .
```

```python
from src.models.linear_integration.li_net3.li_net import LINet
from src.models.linear_integration.li_net3.blocks import LIBasicBlock

# Create a 2-stream LINet (RGB + Depth) with ResNet-18 backbone
model = LINet(
    block=LIBasicBlock,
    layers=[2, 2, 2, 2],          # ResNet-18 layer config
    num_classes=19,                # SUN RGB-D scene categories
    stream_input_channels=[3, 1], # RGB (3ch) + Depth (1ch)
)

# Forward pass expects a list of tensors, one per stream
rgb = torch.randn(4, 3, 224, 224)    # batch of RGB images
depth = torch.randn(4, 1, 224, 224)  # batch of depth maps
outputs = model([rgb, depth])         # (4, 19) class logits
```

Training notebooks for Google Colab are in `notebooks/`.

## Project Structure

```
Multi-Stream-Neural-Networks/
├── src/
│   ├── models/
│   │   ├── linear_integration/li_net3/  # LINet implementation
│   │   ├── multi_channel/               # MCResNet baseline
│   │   ├── core/                        # Base ResNet
│   │   ├── abstracts/                   # Abstract model base class
│   │   └── common/                      # Shared utilities
│   ├── data_utils/                      # Dataset loaders (SUN RGB-D, ScanNet)
│   ├── training/                        # Training infrastructure
│   │   ├── schedulers.py                # 20+ LR scheduler variants
│   │   ├── optimizers.py                # Stream-specific learning rates
│   │   ├── gpu_augmentation.py          # GPU-side augmentation
│   │   ├── modality_dropout.py          # Modality dropout
│   │   └── callbacks/                   # Gradient, weight, pathway monitoring
│   └── evaluation/                      # Ablation, robustness, pathway analysis
├── configs/                             # YAML configs (data, model, experiment)
├── notebooks/                           # Colab training notebooks
├── scripts/                             # Preprocessing and utility scripts
├── tests/                               # Test suite
└── docs/                                # Research documents and design docs
```

## Dataset: SUN RGB-D

LINet is evaluated on the **SUN RGB-D** benchmark (Song et al. 2015) using the standard 19-category scene classification task:

- **9,504 RGB-D images** across 19 scene categories
- **Official train/test split**: 4,845 train / 4,659 test
- Preprocessed to 256x256 tensors, cropped to 224x224 during training

Preprocessing: `python3 scripts/preprocess_sunrgbd_19.py --data-root <path> --no-val-split`

## Future Work

- **NLP extension**: Applying multi-stream integration to text classification with semantic, phonetic, and morphological representations ([proposal](docs/NLP_MultiStream_Proposal.md))
- **Ongoing hyperparameter optimization** and training recipe refinement

## Research Documents

- [MSNN Research Proposal](docs/MSNN_Research_Proposal.md) -- Project goals and biological inspiration
- [Integration Mechanism Research](docs/Integration_Mechanism_Research.md) -- Core research framework
- [LINet Design Plan](docs/LINet_Design_Plan.md) -- Architecture design decisions
- [GPU Optimization Results](docs/gpu_optimization_results.md) -- Performance profiling and optimization
- [NLP Multi-Stream Proposal](docs/NLP_MultiStream_Proposal.md) -- Extension to natural language processing

## License

MIT License -- see LICENSE file for details.
