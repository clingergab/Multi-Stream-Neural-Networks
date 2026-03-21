# MSNN Architecture: Mapping CNN Neurons to Multi-Stream Neurons

## How a Standard CNN Neuron Works

In a traditional CNN, a single "neuron" performs three steps:

1. **Filter (weights):** A small kernel (e.g. 3x3) slides across the input, computing a weighted sum at each spatial position.
2. **Bias:** A scalar bias is added to the result.
3. **Activation:** The result passes through a nonlinear function (e.g. ReLU).

```
output = activation(conv2d(input, filter) + bias)
```

A convolutional layer is a collection of these neurons — one per output filter — each learning different weights to detect different features. The output of the layer is a stack of 2D feature maps.

## How an MSNN Neuron Works (LIConv2d)

Our core idea: **each neuron has a separate, dedicated filter (weight) for each input modality**. Instead of one filter seeing a single fused input, the neuron maintains independent pathways that are integrated before activation.

A single MSNN neuron (one output channel of `LIConv2d`) performs:

### 1. Dendritic Filtering — Separate filter per modality

Each stream/modality has its own learned conv filter. They process their inputs independently:

```
dendritic_signal[i] = conv2d(input[i], filter[i])    # for each modality i
```

In a 2-stream model (e.g. RGB + Depth), there are 2 separate 3x3 filters per neuron. Each filter learns to extract features from its specific modality.

### 2. Soma Integration — Combine before activation

The raw filtered signals from all modalities are combined via learned 1x1 mixing weights, along with the previous integrated state:

```
integrated = W_prev * prev_integrated + sum(W_i * dendritic_signal[i]) + integrated_bias
```

This is the key: integration happens on the **raw conv outputs** (before stream biases or activation), so the neuron sees the pure filtered signal from each modality before deciding how to combine them.

### 3. Activation — After integration

```
output = relu(integrated)
```

The activation function fires on the already-integrated signal, matching our design intent: **integrate first, then activate**.

### 4. Stream Outputs — Parallel independent pathways

Each stream also maintains its own independent output (with its own bias and activation), so modality-specific representations are preserved alongside the integrated one:

```
stream_output[i] = relu(dendritic_signal[i] + stream_bias[i])
```

## Side-by-Side Comparison

| Concept | Standard CNN | MSNN (LIConv2d) |
|---|---|---|
| Weights (filters) | 1 filter per neuron | N filters per neuron (1 per modality) |
| What the filter sees | Single input tensor | One modality's input only |
| Integration | N/A (single input) | Learned 1x1 mixing of all streams |
| When integration happens | N/A | After filtering, before activation |
| Activation | On single conv output | On integrated signal |
| Output | 1 feature map per filter | N stream feature maps + 1 integrated |

## Biological Analogy

- **Dendrites** = The separate per-stream conv filters. Each dendrite receives input from one modality and does its own spatial filtering.
- **Soma** = The integration step. The cell body combines all dendritic signals using learned weights and applies its own firing threshold (integrated bias).
- **Axon output** = The post-activation integrated signal, passed to the next layer.
- **Stream biases** = Pathway-specific baseline potentials, independent from the soma's threshold.

## Where This Lives in Code

- **Separate filters:** `stream_weights` in `_LIConvNd.__init__` — one `Parameter` per stream, each with full kernel size (e.g. 3x3).
- **Integration weights:** `integration_from_streams` — learned 1x1 convs that mix each stream's raw output into the integrated signal.
- **Integrated pathway:** `integrated_weight` — a 1x1 conv that carries forward the previous layer's integrated representation.
- **Forward flow:** `LIConv2d._conv_forward` — filters each stream independently, then integrates raw outputs before bias/activation.
- **Block-level activation:** `LIBasicBlock.forward` / `LIBottleneck.forward` — BN + ReLU applied after the conv+integration step.

## Key Design Decisions

1. **Integration uses raw conv outputs (no stream bias).** Stream biases are pathway-specific thresholds; the soma has its own separate threshold (`integrated_bias`). This prevents double-counting biases during integration.

2. **Integration weights are 1x1 convolutions.** They mix channels but don't do spatial filtering — spatial feature extraction is the job of the per-stream filters.

3. **The integrated pathway also receives its own previous state** via a 1x1 conv (`integrated_weight`), creating a residual-like path for the integrated representation across layers.
