# Modality Dropout Implementation Plan

## Overview

Implement **per-sample** modality dropout to train LINet3 to handle missing streams. Each sample in a batch can independently have a different stream blanked (or no stream blanked). When a stream is blanked for a sample, that sample's contribution should be invisible: no BN stats affected, no gradients, no contribution to integration.

## Key Design Decision

**Approach**: Per-sample masking with subset-based BatchNorm.

1. **Conv**: Compute all samples → mask blanked samples to zero
2. **BN**: Index active samples → standard BN on subset → scatter back (zeros stay zeros)
3. **Other layers**: No special handling (ReLU(0)=0, MaxPool(zeros)=zeros)

This eliminates the need for custom masked BN logic.

## Zero Propagation Through Network

Verify the masking propagates correctly from input to layer1:

1. **conv1**: Masks outputs → blanked samples have `stream_out = 0`, `stream_out_raw = 0`
2. **bn1**: Subset BN → zeros stay zeros (blanked samples skipped)
3. **relu**: ReLU(0) = 0 ✓
4. **maxpool**: MaxPool(zeros) = zeros ✓
5. **layer1**: Receives zeros for blanked stream inputs ✓

**Residual connection correctness**: When a block receives zeros for blanked samples:
- `stream_identities = stream_inputs.copy()` → already zeros for blanked samples
- Block processes → outputs zeros for blanked samples
- `stream_outputs = [s + s_id for s, s_id in zip(stream_outputs, stream_identities)]` → 0 + 0 = 0 ✓

## Mask Format

```python
# blanked_mask: Optional[dict[int, Tensor]]
# Key: stream index (0 = RGB, 1 = Depth for 2-stream)
# Value: bool tensor of shape [batch_size] where True = sample is blanked for this stream

# Example for batch_size=8, 2 streams:
blanked_mask = {
    0: tensor([False, True, False, False, True, False, False, False]),  # samples 1,4 have RGB blanked
    1: tensor([False, False, True, False, False, False, True, False]),  # samples 2,6 have Depth blanked
}
# samples 0,3,5,7 have no stream blanked (normal processing)
# RULE: Never True for both streams on same sample
```

---

## Files to Modify

### 1. `src/training/modality_dropout.py` (NEW FILE)

**Create new file with modality dropout utilities**:

```python
"""
Modality dropout utilities for training multi-stream networks.

Provides per-sample modality dropout to train models to handle missing streams.
"""

import torch
from typing import Optional


def get_modality_dropout_prob(
    epoch: int,
    start_epoch: int = 0,
    ramp_epochs: int = 20,
    final_rate: float = 0.2
) -> float:
    """
    Compute modality dropout probability based on current epoch.

    Schedule:
    - Before start_epoch: 0% dropout
    - During ramp (start_epoch to start_epoch + ramp_epochs): Linear ramp 0% → final_rate
    - After ramp: final_rate

    Args:
        epoch: Current training epoch (0-indexed)
        start_epoch: Epoch to start dropout (default: 0)
        ramp_epochs: Number of epochs to ramp from 0 to final_rate (default: 20)
        final_rate: Final dropout probability (default: 0.2 = 20%)

    Returns:
        Dropout probability for this epoch
    """
    if epoch < start_epoch:
        return 0.0
    epochs_since_start = epoch - start_epoch
    ramp_epochs = max(1, ramp_epochs)  # Prevent division by zero
    if epochs_since_start < ramp_epochs:
        return final_rate * (epochs_since_start / ramp_epochs)
    return final_rate


def generate_per_sample_blanked_mask(
    batch_size: int,
    num_streams: int,
    dropout_prob: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None  # For reproducibility
) -> Optional[dict[int, torch.Tensor]]:
    """
    Generate per-sample blanked mask for modality dropout.

    Rules:
    - Each sample can have at most ONE stream blanked
    - Never blank both streams for the same sample
    - dropout_prob is the per-sample probability of blanking ANY stream

    Args:
        batch_size: Number of samples in batch
        num_streams: Number of input streams (e.g., 2 for RGB+Depth)
        dropout_prob: Per-sample probability of blanking a stream
        device: Device to create tensors on
        generator: Optional random generator for reproducibility

    Returns:
        None if dropout_prob <= 0 or no samples selected for dropout
        dict mapping stream_idx -> bool tensor [batch_size] where True = blanked
    """
    if dropout_prob <= 0.0:
        return None

    # Step 1: Decide which samples get a stream blanked
    should_blank = torch.rand(batch_size, device=device, generator=generator) < dropout_prob

    # Step 2: For blanked samples, randomly pick which stream to blank
    stream_to_blank = torch.randint(0, num_streams, (batch_size,), device=device, generator=generator)

    # Step 3: Create per-stream masks
    blanked_mask = {}
    for stream_idx in range(num_streams):
        blanked_mask[stream_idx] = should_blank & (stream_to_blank == stream_idx)

    # Return None if no samples are blanked (optimization)
    if not any(mask.any() for mask in blanked_mask.values()):
        return None

    return blanked_mask
```

### 2. `src/models/linear_integration/li_net3/conv.py`

> Note: No changes to schedulers.py - all modality dropout code is in the new modality_dropout.py file.

**Modify `LIConv2d._conv_forward()`** - mask after conv:

```python
def _conv_forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor],
    stream_weights: list[Tensor],
    integrated_weight: Optional[Tensor],
    integration_from_streams_weights: list[Tensor],
    stream_biases: list[Optional[Tensor]],
    integrated_bias: Optional[Tensor],
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW
) -> tuple[list[Tensor], Tensor]:
    stream_outputs = []
    stream_outputs_raw = []

    for i, (stream_input, stream_weight, stream_bias) in enumerate(
        zip(stream_inputs, stream_weights, stream_biases)
    ):
        # Normal conv computation for all samples
        if self.padding_mode != "zeros":
            stream_out_raw = F.conv2d(
                F.pad(stream_input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                stream_weight, None, self.stride, _pair(0), self.dilation, self.groups,
            )
        else:
            stream_out_raw = F.conv2d(
                stream_input, stream_weight, None, self.stride, self.padding, self.dilation, self.groups
            )

        # Add bias for stream output
        if stream_bias is not None:
            stream_out = stream_out_raw + stream_bias.view(1, -1, 1, 1)
        else:
            stream_out = stream_out_raw

        # === PER-SAMPLE MASKING: Zero blanked samples ===
        stream_blanked = blanked_mask.get(i) if blanked_mask else None
        if stream_blanked is not None and stream_blanked.any():
            # mask: [batch_size, 1, 1, 1] for broadcasting, 1.0 for active, 0.0 for blanked
            mask = (~stream_blanked).float().view(-1, 1, 1, 1)
            stream_out = stream_out * mask
            stream_out_raw = stream_out_raw * mask  # Critical: zeros for integration

        stream_outputs.append(stream_out)
        stream_outputs_raw.append(stream_out_raw)

    # Integration unchanged - stream_outputs_raw already has zeros for blanked samples
    # ... existing integration code ...
```

**Modify `LIConv2d.forward()`**:
- Add `blanked_mask` parameter, pass to `_conv_forward`

**Modify `LIBatchNorm2d.forward()`** - subset BN approach:

```python
def forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor] = None,
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW
) -> tuple[list[Tensor], Tensor]:
    # Check input dimensions
    for stream_input in stream_inputs:
        self._check_input_dim(stream_input)
    if integrated_input is not None:
        self._check_input_dim(integrated_input)

    # Compute exponential_average_factor ONCE
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        if self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum

    # Process all stream pathways
    stream_outputs = []
    for i, stream_input in enumerate(stream_inputs):
        stream_blanked = blanked_mask.get(i) if blanked_mask else None

        if stream_blanked is None or not stream_blanked.any():
            # No blanking - standard BN on all samples
            stream_out = self._forward_single_pathway(
                stream_input,
                getattr(self, f"stream{i}_running_mean"),
                getattr(self, f"stream{i}_running_var"),
                self.stream_weights[i] if self.affine else None,
                self.stream_biases[i] if self.affine else None,
                exponential_average_factor,
            )
        elif stream_blanked.all():
            # All samples blanked - pass through zeros unchanged
            stream_out = stream_input  # Already zeros from conv masking
        else:
            # Partial blanking - subset BN on active samples only
            active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
            active_input = stream_input[active_idx]  # [num_active, C, H, W]

            # Standard BN on active samples - this updates running stats correctly
            active_output = self._forward_single_pathway(
                active_input,
                getattr(self, f"stream{i}_running_mean"),
                getattr(self, f"stream{i}_running_var"),
                self.stream_weights[i] if self.affine else None,
                self.stream_biases[i] if self.affine else None,
                exponential_average_factor,
            )

            # Scatter back - blanked samples stay as zeros
            stream_out = torch.zeros_like(stream_input)
            stream_out[active_idx] = active_output

        stream_outputs.append(stream_out)

    # Process integrated pathway (no masking - always all samples)
    if integrated_input is not None:
        integrated_out = self._forward_single_pathway(
            integrated_input,
            self.integrated_running_mean,
            self.integrated_running_var,
            self.integrated_weight,
            self.integrated_bias,
            exponential_average_factor,
        )
    else:
        integrated_out = None

    return stream_outputs, integrated_out
```

### 3. `src/models/linear_integration/li_net3/container.py`

**Modify `LISequential.forward()`**:
```python
def forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor] = None,
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW
) -> tuple[list[Tensor], Tensor]:
    for module in self:
        stream_inputs, integrated_input = module(stream_inputs, integrated_input, blanked_mask)
    return stream_inputs, integrated_input
```

**Modify `LIReLU.forward()`**:
```python
def forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor] = None,
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW - no action needed
) -> tuple[list[Tensor], Tensor]:
    # ReLU doesn't need special handling: ReLU(0) = 0 ✓
    stream_outputs = [F.relu(s, inplace=self.inplace) for s in stream_inputs]
    integrated_out = F.relu(integrated_input, inplace=self.inplace) if integrated_input is not None else None
    return stream_outputs, integrated_out
```

### 4. `src/models/linear_integration/li_net3/pooling.py`

**Modify `LIMaxPool2d.forward()`**:
```python
def forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor] = None,
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW - no action needed
) -> tuple[list[Tensor], Tensor]:
    # MaxPool doesn't need special handling: MaxPool(zeros) = zeros ✓
    stream_outputs = [F.max_pool2d(s, self.kernel_size, self.stride,
                                    self.padding, self.dilation, self.ceil_mode)
                      for s in stream_inputs]
    integrated_out = F.max_pool2d(integrated_input, ...) if integrated_input is not None else None
    return stream_outputs, integrated_out
```

### 5. `src/models/linear_integration/li_net3/blocks.py`

**Modify `LIBasicBlock.forward()`**:
```python
def forward(
    self,
    stream_inputs: list[Tensor],
    integrated_input: Optional[Tensor] = None,
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW
) -> tuple[list[Tensor], Tensor]:
    stream_identities = stream_inputs.copy()
    integrated_identity = integrated_input

    # Propagate mask through all layers
    stream_outputs, integrated = self.conv1(stream_inputs, integrated_input, blanked_mask)
    stream_outputs, integrated = self.bn1(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.relu(stream_outputs, integrated, blanked_mask)

    stream_outputs, integrated = self.conv2(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.bn2(stream_outputs, integrated, blanked_mask)

    # Downsample is an LISequential containing LIConv2d + LIBatchNorm2d
    # LISequential.forward() accepts blanked_mask and propagates it to all contained modules
    if self.downsample is not None:
        stream_identities, integrated_identity = self.downsample(
            stream_identities, integrated_identity, blanked_mask)

    # Residual connections
    # Note: stream_inputs arrived already zeros for blanked samples (from previous layer)
    # So stream_identities is zeros too, and stream_outputs is zeros
    # Result: 0 + 0 = 0 for blanked samples ✓
    stream_outputs = [s + s_id for s, s_id in zip(stream_outputs, stream_identities)]
    if integrated_identity is not None:
        integrated = integrated + integrated_identity

    stream_outputs, integrated = self.relu(stream_outputs, integrated, blanked_mask)
    return stream_outputs, integrated
```

**Modify `LIBottleneck.forward()`** - same pattern (propagate blanked_mask to all layers).

**Note on Downsample**: The `downsample` attribute is an `LISequential` containing `LIConv2d` and `LIBatchNorm2d`. Since we're modifying `LISequential.forward()` to accept and propagate `blanked_mask`, the downsample path will automatically handle masking correctly.

### 6. `src/models/linear_integration/li_net3/li_net.py`

**Modify `forward()`**:
```python
def forward(
    self,
    stream_inputs: list[Tensor],
    blanked_mask: Optional[dict[int, Tensor]] = None  # NEW
) -> Tensor:
    # Propagate mask through entire network
    stream_outputs, integrated = self.conv1(stream_inputs, None, blanked_mask)
    stream_outputs, integrated = self.bn1(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.relu(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.maxpool(stream_outputs, integrated, blanked_mask)

    stream_outputs, integrated = self.layer1(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.layer2(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.layer3(stream_outputs, integrated, blanked_mask)
    stream_outputs, integrated = self.layer4(stream_outputs, integrated, blanked_mask)

    # Classification (unchanged)
    integrated_pooled = self.avgpool(integrated)
    integrated_features = torch.flatten(integrated_pooled, 1)
    integrated_features = self.dropout(integrated_features)
    logits = self.fc(integrated_features)
    return logits
```

**Modify `fit()`** - add parameters and schedule usage:
```python
# Add to imports at top of file:
from src.training.modality_dropout import get_modality_dropout_prob, generate_per_sample_blanked_mask

# Add parameters to fit():
modality_dropout: bool = False,
modality_dropout_start: int = 0,
modality_dropout_ramp: int = 20,
modality_dropout_rate: float = 0.2,

# Add history tracking in fit():
if modality_dropout:
    history['modality_dropout_prob'] = []

# In the epoch loop, compute dropout prob and pass to _train_epoch():
for epoch in range(epochs):
    # Compute modality dropout probability for this epoch
    modality_dropout_prob = 0.0
    if modality_dropout:
        modality_dropout_prob = get_modality_dropout_prob(
            epoch=epoch,
            start_epoch=modality_dropout_start,
            ramp_epochs=modality_dropout_ramp,
            final_rate=modality_dropout_rate
        )
        history['modality_dropout_prob'].append(modality_dropout_prob)

    # Pass to _train_epoch
    avg_train_loss, train_accuracy, stream_train_accs = self._train_epoch(
        train_loader, history, pbar, gradient_accumulation_steps, grad_clip_norm,
        clear_cache_per_epoch, stream_monitoring=stream_monitoring,
        aux_optimizer=aux_optimizer,
        modality_dropout_prob=modality_dropout_prob  # NEW
    )
```

**Modify `_train_epoch()`**:
```python
def _train_epoch(self, ..., modality_dropout_prob: float = 0.0):
    self.train()
    ...

    # Change: Per-stream active sample counters (for accurate accuracy with dropout)
    stream_train_correct = [0] * self.num_streams
    stream_train_active = [0] * self.num_streams  # CHANGED from stream_train_total

    for batch_idx, batch_data in enumerate(train_loader):
        *stream_batches, targets = batch_data
        stream_batches = [b.to(self.device, non_blocking=True) for b in stream_batches]
        targets = targets.to(self.device, non_blocking=True)

        # Generate per-sample blanked mask
        blanked_mask = None
        if modality_dropout_prob > 0:
            blanked_mask = generate_per_sample_blanked_mask(
                batch_size=targets.shape[0],
                num_streams=self.num_streams,
                dropout_prob=modality_dropout_prob,
                device=self.device
            )

        # Forward pass with mask
        outputs = self(stream_batches, blanked_mask=blanked_mask)
        loss = self.criterion(outputs, targets)
        loss.backward()
        ...

    # Final accuracy calculation (per-stream active samples)
    stream_accs = [
        stream_train_correct[i] / max(stream_train_active[i], 1) if stream_monitoring else 0.0
        for i in range(self.num_streams)
    ]
```

**Modify `_validate()`** - NO mask during validation (unchanged).

**Modify `evaluate()`** - add optional blanked_streams for robustness testing:
```python
def evaluate(
    self,
    data_loader: DataLoader,
    stream_monitoring: bool = True,
    blanked_streams: Optional[set[int]] = None  # For single-stream robustness eval
) -> dict[str, float]:
    """
    Evaluate model. Use blanked_streams to test single-stream robustness.

    Args:
        blanked_streams: Set of stream indices to blank for ALL samples.
                        {0} = blank RGB, test depth-only
                        {1} = blank depth, test RGB-only
    """
    ...
    with torch.no_grad():
        for batch_data in data_loader:
            ...
            # Convert blanked_streams set to blanked_mask for entire batch
            blanked_mask = None
            if blanked_streams:
                batch_size = stream_batches[0].shape[0]
                blanked_mask = {
                    i: torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    if i in blanked_streams else
                    torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    for i in range(self.num_streams)
                }

            outputs = self(stream_batches, blanked_mask=blanked_mask)
            ...
```

---

## Implementation Order

1. **Create `src/training/modality_dropout.py`** - new file with `get_modality_dropout_prob()` and `generate_per_sample_blanked_mask()`
2. **Modify `conv.py`** - `LIConv2d._conv_forward()` and `forward()` with per-sample masking after conv
3. **Modify `conv.py`** - `LIBatchNorm2d.forward()` with subset BN approach
4. **Modify `container.py`** - `LISequential.forward()` and `LIReLU.forward()` to propagate mask
5. **Modify `pooling.py`** - `LIMaxPool2d.forward()` to propagate mask
6. **Modify `blocks.py`** - `LIBasicBlock.forward()` and `LIBottleneck.forward()` to propagate mask
7. **Modify `li_net.py`** - `forward()`, `fit()`, `_train_epoch()`, and `evaluate()`
8. **Test end-to-end** - verify mask generation, zero propagation, and gradient isolation

---

## Usage Examples

### Training with Per-Sample Modality Dropout
```python
history = model.fit(
    train_loader,
    val_loader=val_loader,
    epochs=100,
    modality_dropout=True,
    modality_dropout_start=20,   # Start at epoch 20
    modality_dropout_ramp=20,    # Ramp over 20 epochs
    modality_dropout_rate=0.2,   # 20% of samples have a stream blanked
)
```

### Single-Stream Robustness Evaluation
```python
# Normal (all streams)
results = model.evaluate(test_loader)
print(f"Both streams: {results['accuracy']:.4f}")

# RGB only (depth blanked for all samples)
results_rgb = model.evaluate(test_loader, blanked_streams={1})
print(f"RGB only: {results_rgb['accuracy']:.4f}")

# Depth only (RGB blanked for all samples)
results_depth = model.evaluate(test_loader, blanked_streams={0})
print(f"Depth only: {results_depth['accuracy']:.4f}")
```

---

## Key Design Points

### Why Mask in Both Conv AND BN?

1. **Conv masking**: Zeros the raw conv output for blanked samples
   - Critical because BN stats are computed from conv outputs
   - If we didn't mask here, BN would include blanked sample features in stats

2. **BN subset indexing**: Only processes active samples
   - Standard BN would normalize zeros → non-zeros (due to γ/β)
   - Subset approach: BN only sees real data, blanked samples stay zero

### Why Not Custom Masked BN?

The custom `_forward_single_pathway_masked` approach had a bug: BN turns zeros back to non-zeros. The subset approach is simpler and uses standard, well-tested BN.

### Gradient Isolation

When we multiply by mask in conv:
```python
stream_out_raw = stream_out_raw * mask  # mask is 0 for blanked samples
```

During backprop:
```
d(loss)/d(stream_out_raw) = d(loss)/d(masked_out) * mask = 0 for blanked
```

No gradient flows to blanked stream's weights for blanked samples.

### Performance Notes

- Conv is computed for all samples (no indexing overhead there)
- BN subset indexing adds minor overhead but uses highly optimized standard BN
- For 20% dropout: ~90% samples are active per stream → minimal overhead

---

## Stream Monitoring Considerations

**Issue**: Stream monitoring uses `_forward_stream_pathway()` which processes the original input data. When modality dropout blanks a stream for certain samples, stream monitoring would still process those samples with full input data - creating a mismatch.

**Solution**: Skip blanked samples in stream monitoring. When computing:
1. **Auxiliary classifier training**: Only include loss for non-blanked samples
2. **Accuracy measurement**: Only count samples where the stream was NOT blanked

**Implementation in `_train_epoch()`**:

**1. Initialize per-stream counters at function start** (replace existing `stream_train_total`):
```python
def _train_epoch(self, ..., modality_dropout_prob: float = 0.0):
    self.train()
    train_loss = 0.0
    train_batches = 0
    train_correct = 0
    train_total = 0

    # Stream-specific tracking (if monitoring enabled) - N streams
    stream_train_correct = [0] * self.num_streams
    stream_train_active = [0] * self.num_streams  # CHANGED: per-stream active sample count (was stream_train_total)
```

**2. In the stream monitoring section**, handle blanked samples:
```python
if stream_monitoring:
    # ... existing eval mode switch ...

    # Forward through stream pathways and compute auxiliary losses
    aux_losses = []
    for i in range(self.num_streams):
        # Get mask for this stream
        stream_blanked = blanked_mask.get(i) if blanked_mask else None

        # Forward through this stream's pathway
        stream_features = self._forward_stream_pathway(i, stream_batches[i])
        stream_features_detached = stream_features.detach()
        stream_outputs = self.fc_streams[i](stream_features_detached)

        # Compute auxiliary loss ONLY for non-blanked samples
        if stream_blanked is not None and stream_blanked.any():
            # Get indices of active (non-blanked) samples
            active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
            if len(active_idx) > 0:
                aux_loss = self.criterion(stream_outputs[active_idx], targets[active_idx])
                aux_losses.append(aux_loss)
            # If all samples blanked for this stream, skip this stream's aux loss
        else:
            # No blanking - use all samples
            aux_loss = self.criterion(stream_outputs, targets)
            aux_losses.append(aux_loss)

    # ... backward pass only on collected aux_losses ...

    # === ACCURACY MEASUREMENT PHASE ===
    with torch.no_grad():
        self.eval()
        for i in range(self.num_streams):
            stream_features = self._forward_stream_pathway(i, stream_batches[i])
            stream_outputs = self.fc_streams[i](stream_features)
            stream_pred = stream_outputs.argmax(1)

            # Only count non-blanked samples for accuracy
            stream_blanked = blanked_mask.get(i) if blanked_mask else None
            if stream_blanked is not None and stream_blanked.any():
                active_idx = (~stream_blanked).nonzero(as_tuple=True)[0]
                if len(active_idx) > 0:
                    stream_train_correct[i] += (stream_pred[active_idx] == targets[active_idx]).sum().item()
                    stream_train_active[i] += len(active_idx)
            else:
                stream_train_correct[i] += (stream_pred == targets).sum().item()
                stream_train_active[i] += targets.size(0)

        self.train(was_training)  # Restore training mode
```

**3. Final accuracy calculation** at end of `_train_epoch()`:
```python
# Calculate stream-specific accuracies (N streams)
stream_accs = [
    stream_train_correct[i] / max(stream_train_active[i], 1) if stream_monitoring else 0.0
    for i in range(self.num_streams)
]
```

**Summary of stream monitoring behavior with modality dropout:**
- **Loss**: Only non-blanked samples contribute to auxiliary loss for each stream
- **Accuracy**: Only non-blanked samples counted (per-stream active counters)
- **Result**: Stream monitoring metrics reflect actual stream performance, unaffected by dropout noise

---

## Verification

1. **Mask generation test**: Verify never both streams blanked for same sample
2. **Conv masking test**: Verify blanked samples have zero stream_out_raw
3. **BN subset test**: Verify BN running stats only include active samples
4. **Zero propagation test**: Verify zeros propagate through ReLU, MaxPool correctly
5. **Gradient test**: Verify blanked stream weights get zero gradient for blanked samples
6. **Integration weights gradient test**: Verify integration weights receive gradients from non-blanked samples
7. **Validation no dropout test**: Verify `_validate()` doesn't apply modality dropout (blanked_mask=None)
8. **History tracking test**: Verify `history['modality_dropout_prob']` is populated correctly with expected schedule values
9. **Stream monitoring isolation test**: Verify blanked samples are excluded from stream accuracy metrics
10. **Integration test**: Full training run with modality_dropout=True
11. **Single-stream eval test**: Test `evaluate()` with blanked_streams parameter
