# Multi-Stream Neural Networks

Biologically-inspired neural network architectures for integrating complementary data streams. This research explores how biological neurons integrate information and applies these principles to computer vision and natural language processing.

## Overview

**Core Question**: What is the optimal way to integrate multiple streams of information in neural networks?

Most multi-modal AI systems use ad-hoc fusion strategies. In contrast, biological neurons appear to use a universal integration mechanism across all brain regions. This project investigates whether we can discover and implement that mechanism in artificial neural networks.

**Key Principle**: Real neurons integrate signals from multiple dendritic inputs *before* applying a firing threshold—not after. This integration-before-threshold approach enables discovering optimal combinations of complementary features.

## Current Work

### Computer Vision: RGB-D Scene Classification
- **Task**: Scene classification on SUN-RGBD dataset
- **Streams**: RGB (color) + Depth (distance)
- **Backbone**: ResNet-18
- **Status**: ✅ Implemented and training

### Natural Language Processing: Multi-Representation Text Classification
- **Task**: Sentiment analysis on SST-5 dataset
- **Streams**: Semantic (meaning) + Phonetic (sound) + Morphological (structure)
- **Backbone**: BERT
- **Status**: 🔄 Proposed ([see details](docs/NLP_MultiStream_Proposal.md))

## Integration Architectures

Four integration mechanisms implemented, ranging from high expressivity to maximum biological plausibility:

| Architecture | Integration Method | Normalization | Biological Plausibility |
|-------------|-------------------|---------------|------------------------|
| **LINet3** | Full weight matrices | BN on integrated | Moderate |
| **LINet3-Soma** | Full weight matrices | None (raw integration) | High |
| **DMNet** | Scalar weights | None (raw integration) | High |
| **DMNet-Conv** | Scalar weights + Conv1x1 recurrent | None (raw integration) | Highest |

All architectures integrate *raw* stream outputs (without biases) before applying a threshold, mirroring biological dendritic integration.

## Quick Start

```bash
# Installation
git clone https://github.com/your-username/Multi-Stream-Neural-Networks.git
cd Multi-Stream-Neural-Networks
pip install -e .
```

```python
# Create a multi-stream model
from src.models.linear_integration.li_net3.li_net import create_linet3_resnet18

model = create_linet3_resnet18(
    num_classes=19,  # SUN-RGBD scene categories
    num_streams=2,   # RGB + Depth
    pretrained=True
)

# Forward pass with two streams
outputs = model([rgb_tensor, depth_tensor])
```

## Research Documents

- **[Integration Mechanism Research](docs/Integration_Mechanism_Research.md)**: Core research framework and experimental protocol
- **[MSNN Research Proposal](docs/MSNN_Research_Proposal.md)**: Overall project goals and biological inspiration
- **[NLP Multi-Stream Proposal](docs/NLP_MultiStream_Proposal.md)**: Extension to natural language processing


## Project Structure

```
Multi-Stream-Neural-Networks/
├── src/models/
│   ├── linear_integration/
│   │   ├── li_net3/          # Linear Integration with BN
│   │   └── li_net3_soma/     # Linear Integration, no BN (biologically plausible)
│   ├── direct_mixing/         # Direct Mixing with scalar weights
│   └── direct_mixing_conv/    # Direct Mixing + Conv1x1 (most biologically plausible)
├── docs/                      # Research proposals and documentation
├── notebooks/                 # Training experiments
└── tests/                     # Test suites
```

## Contributing

This is an active research project. Contributions, suggestions, and collaborations are welcome!

## License

MIT License - see LICENSE file for details.
