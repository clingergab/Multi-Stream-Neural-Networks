# LINet3 N-Stream Refactoring - Implementation Plan

## Executive Summary

Refactor the existing LINet3 architecture from hardcoded 2-stream (RGB + Depth) to flexible N-stream support (RGB + Depth + Orthogonal + future streams). This will enable adding new data streams without modifying the core architecture code.

**Target:** Support 3 streams initially (RGB + Depth + Orthogonal), but design for arbitrary N streams.

**Current State:** All code is hardcoded for 2 streams using individual parameters (stream1_*, stream2_*)
**Target State:** Dynamic N-stream support using `nn.ParameterList` and `List[Tensor]` signatures

**Note**: Factory functions are named `li_resnet18`, `li_resnet34`, etc. (not `li_net*`)

---

## Current Architecture Analysis

### Current 2-Stream Design

**Inputs:**
- Stream 1: RGB (3 channels)
- Stream 2: Depth (1 channel)
- Integrated: Combined representation (starts at layer 2)

**Integration Formula:**
```
integrated_output = W_prev · integrated_prev + W1 · stream1_out + W2 · stream2_out + bias
```

**Key Characteristics:**
- Each stream has independent processing pathway
- Integration weights are 1×1 convolutions (channel-wise only)
- Stream weights are full kernels (3×3, 7×7, etc.)
- Integrated stream accumulates information across network depth
- Final classification uses integrated stream only

### Current Parameter Structure (Example: First Conv Layer)

```python
class LIConv2d:
    def __init__(self,
                 stream1_in_channels=3,      # RGB input
                 stream2_in_channels=1,      # Depth input
                 integrated_in_channels=0,   # No prev integrated (first layer)
                 stream1_out_channels=64,
                 stream2_out_channels=64,
                 integrated_out_channels=64,
                 kernel_size=7,
                 stride=2):

        # Stream-specific weights (full kernels)
        self.stream1_weight = Parameter(torch.empty(64, 3, 7, 7))     # 9,408 params
        self.stream2_weight = Parameter(torch.empty(64, 1, 7, 7))     # 3,136 params

        # Previous integrated stream (1×1 kernel)
        self.integrated_weight = Parameter(torch.empty(64, 0, 1, 1))  # 0 params (first layer)

        # Integration from stream outputs (1×1 kernels)
        self.integration_from_stream1 = Parameter(torch.empty(64, 64, 1, 1))  # 4,096 params
        self.integration_from_stream2 = Parameter(torch.empty(64, 64, 1, 1))  # 4,096 params
```

**Total: ~20,736 parameters** (for first conv layer)

---

## Proposed N-Stream Design

### Target Parameter Structure (N=3 example)

```python
class LIConv2d:
    def __init__(self,
                 stream_in_channels: List[int],      # [3, 1, 1] for RGB, Depth, Orth
                 stream_out_channels: List[int],     # [64, 64, 64]
                 integrated_in_channels: int,        # 0 for first layer, 64 for later
                 integrated_out_channels: int,       # 64
                 kernel_size: int = 3,
                 stride: int = 1):

        num_streams = len(stream_in_channels)

        # Stream-specific weights (full kernels) - DYNAMIC
        self.stream_weights = nn.ParameterList([
            Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
            for in_ch, out_ch in zip(stream_in_channels, stream_out_channels)
        ])
        # stream_weights[0]: 64×3×7×7 = 9,408 params (RGB)
        # stream_weights[1]: 64×1×7×7 = 3,136 params (Depth)
        # stream_weights[2]: 64×1×7×7 = 3,136 params (Orthogonal)

        # Previous integrated stream (1×1 kernel) - UNCHANGED
        if integrated_in_channels > 0:
            self.integrated_weight = Parameter(
                torch.empty(integrated_out_channels, integrated_in_channels, 1, 1)
            )

        # Integration from stream outputs (1×1 kernels) - DYNAMIC
        self.integration_from_streams = nn.ParameterList([
            Parameter(torch.empty(integrated_out_channels, out_ch, 1, 1))
            for out_ch in stream_out_channels
        ])
        # integration_from_streams[0]: 64×64×1×1 = 4,096 params (from RGB)
        # integration_from_streams[1]: 64×64×1×1 = 4,096 params (from Depth)
        # integration_from_streams[2]: 64×64×1×1 = 4,096 params (from Orth)
```

**Total: ~27,968 parameters** (for first conv layer with N=3)
**Increase: +35% vs 2-stream**

### Integration Formula (N-Stream)

```python
# General formula for N streams:
integrated_output = W_prev · integrated_prev + Σ(Wi · stream_i_out for i=0 to N-1) + bias

# Concrete example (N=3):
integrated_output = W_prev · integrated_prev +
                    W0 · rgb_out +
                    W1 · depth_out +
                    W2 · orth_out +
                    bias
```

---

## Implementation Details by File

### 1. conv.py - Core Convolution Layers

#### 1.1 LIConv2d Class

**CURRENT (2-stream) signature:**
```python
def __init__(self,
             stream1_in_channels: int,              # Current: individual int
             stream2_in_channels: int,              # Current: individual int
             integrated_in_channels: int,
             stream1_out_channels: int,             # Current: individual int
             stream2_out_channels: int,             # Current: individual int
             integrated_out_channels: int,
             kernel_size: _size_2_t,
             stride: _size_2_t = 1,
             padding: _size_2_t | str = 0,
             dilation: _size_2_t = 1,
             groups: int = 1,
             bias: bool = True,
             padding_mode: str = 'zeros',
             device=None,
             dtype=None)
```

**TARGET (N-stream) signature:**
```python
def __init__(self,
             stream_in_channels: List[int],          # NEW: [3, 1, 1] for RGB, Depth, Orth
             stream_out_channels: List[int],         # NEW: [64, 64, 64]
             integrated_in_channels: int,            # UNCHANGED
             integrated_out_channels: int,           # UNCHANGED
             kernel_size: _size_2_t,
             stride: _size_2_t = 1,
             padding: _size_2_t | str = 0,
             dilation: _size_2_t = 1,
             groups: int = 1,
             bias: bool = True,
             padding_mode: str = 'zeros',
             device=None,
             dtype=None)
```

**CURRENT (2-stream) parameter structure:**
```python
# Stream-specific weights (full kernels)
self.stream1_weight = Parameter(torch.empty(stream1_out_channels, stream1_in_channels, *kernel_size))
self.stream2_weight = Parameter(torch.empty(stream2_out_channels, stream2_in_channels, *kernel_size))

# Previous integrated stream (1×1 kernel)
if integrated_in_channels > 0:
    self.integrated_weight = Parameter(torch.empty(integrated_out_channels, integrated_in_channels, 1, 1))

# Integration from stream outputs (1×1 kernels)
self.integration_from_stream1 = Parameter(torch.empty(integrated_out_channels, stream1_out_channels, 1, 1))
self.integration_from_stream2 = Parameter(torch.empty(integrated_out_channels, stream2_out_channels, 1, 1))

# Biases (if enabled)
if bias:
    self.stream1_bias = Parameter(torch.empty(stream1_out_channels))
    self.stream2_bias = Parameter(torch.empty(stream2_out_channels))
    self.integrated_bias = Parameter(torch.empty(integrated_out_channels))
```

**TARGET (N-stream) parameter structure:**
```python
self.num_streams = len(stream_in_channels)
self.stream_in_channels = stream_in_channels
self.stream_out_channels = stream_out_channels

# Stream-specific weights (full kernels) - DYNAMIC using ParameterList
self.stream_weights = nn.ParameterList([
    Parameter(torch.empty(
        stream_out_channels[i],
        stream_in_channels[i],
        *self.kernel_size,
        **factory_kwargs
    ))
    for i in range(self.num_streams)
])

# Previous integrated stream (1×1 kernel) - UNCHANGED
if integrated_in_channels > 0:
    self.integrated_weight = Parameter(torch.empty(
        integrated_out_channels,
        integrated_in_channels,
        1, 1,
        **factory_kwargs
    ))

# Integration from stream outputs (1×1 kernels) - DYNAMIC using ParameterList
self.integration_from_streams = nn.ParameterList([
    Parameter(torch.empty(
        integrated_out_channels,
        stream_out_channels[i],
        1, 1,
        **factory_kwargs
    ))
    for i in range(self.num_streams)
])

# Biases (if enabled) - DYNAMIC using ParameterList
if bias:
    self.stream_biases = nn.ParameterList([
        Parameter(torch.empty(stream_out_channels[i], **factory_kwargs))
        for i in range(self.num_streams)
    ])
    self.integrated_bias = Parameter(torch.empty(integrated_out_channels, **factory_kwargs))
else:
    self.stream_biases = None
    self.integrated_bias = None
```

**CURRENT (2-stream) forward method:**
```python
def forward(self,
            stream1_input: Tensor,                       # Individual tensor
            stream2_input: Tensor,                       # Individual tensor
            integrated_input: Optional[Tensor] = None
            ) -> tuple[Tensor, Tensor, Tensor]:          # Returns 3-tuple

    return self._conv_forward(
        stream1_input, stream2_input, integrated_input,
        self.stream1_weight, self.stream2_weight, self.integrated_weight,
        self.stream1_bias, self.stream2_bias, self.integrated_bias
    )
```

**TARGET (N-stream) forward method:**
```python
def forward(self,
            stream_inputs: List[Tensor],                 # List of N stream tensors
            integrated_input: Optional[Tensor] = None
            ) -> Tuple[List[Tensor], Tensor]:            # Returns (stream_outputs, integrated_output)

    # Validate input
    assert len(stream_inputs) == self.num_streams, \
        f"Expected {self.num_streams} streams, got {len(stream_inputs)}"

    # Extract weights and biases as lists
    stream_weights = list(self.stream_weights)
    integration_weights = list(self.integration_from_streams)
    stream_biases = list(self.stream_biases) if self.stream_biases else [None] * self.num_streams

    return self._conv_forward(
        stream_inputs,
        integrated_input,
        stream_weights,
        self.integrated_weight if hasattr(self, 'integrated_weight') else None,
        integration_weights,
        stream_biases,
        self.integrated_bias
    )
```

**_conv_forward implementation:**
```python
# OLD
def _conv_forward(self,
                  stream1_input: Tensor,
                  stream2_input: Tensor,
                  integrated_input: Optional[Tensor],
                  stream1_weight: Tensor,
                  stream2_weight: Tensor,
                  integrated_weight: Tensor,
                  stream1_bias: Optional[Tensor],
                  stream2_bias: Optional[Tensor],
                  integrated_bias: Optional[Tensor]
                  ) -> tuple[Tensor, Tensor, Tensor]:

    # Process stream 1
    stream1_out = F.conv2d(stream1_input, stream1_weight, stream1_bias, ...)

    # Process stream 2
    stream2_out = F.conv2d(stream2_input, stream2_weight, stream2_bias, ...)

    # Integrate
    integrated_from_prev = F.conv2d(integrated_input, integrated_weight, ...)
    integrated_from_s1 = F.conv2d(stream1_out, integration_from_stream1, ...)
    integrated_from_s2 = F.conv2d(stream2_out, integration_from_stream2, ...)
    integrated_out = integrated_from_prev + integrated_from_s1 + integrated_from_s2 + bias

    return stream1_out, stream2_out, integrated_out

# NEW
def _conv_forward(self,
                  stream_inputs: List[Tensor],
                  integrated_input: Optional[Tensor],
                  stream_weights: List[Tensor],
                  integrated_weight: Optional[Tensor],
                  integration_weights: List[Tensor],
                  stream_biases: List[Optional[Tensor]],
                  integrated_bias: Optional[Tensor]
                  ) -> Tuple[List[Tensor], Tensor]:

    # Process each stream independently
    stream_outputs = []
    for i in range(self.num_streams):
        stream_out = F.conv2d(
            stream_inputs[i],
            stream_weights[i],
            stream_biases[i],
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        stream_outputs.append(stream_out)

    # Build integrated stream
    # Start with previous integrated (if exists)
    if integrated_input is not None and integrated_weight is not None:
        integrated_out = F.conv2d(
            integrated_input,
            integrated_weight,
            None,  # No bias yet
            self.stride,  # ⚠️ CRITICAL: Use self.stride (NOT stride=1!) to maintain spatial alignment
            0,  # No padding for 1×1 conv
            1,  # No dilation
            1   # No groups
        )
    else:
        # First layer: no previous integrated stream
        # CORRECTED: Use scalar 0 (broadcasts automatically, more efficient than zeros tensor)
        integrated_out = 0

    # Add contribution from each stream output (1×1 convolutions)
    for i in range(self.num_streams):
        integrated_from_stream_i = F.conv2d(
            stream_outputs[i],
            integration_weights[i],
            None,  # No bias yet
            1,  # Stride=1 for integration
            0,  # No padding for 1×1
            1,  # No dilation
            1   # No groups
        )
        integrated_out = integrated_out + integrated_from_stream_i

    # Add integrated bias
    if integrated_bias is not None:
        integrated_out = integrated_out + integrated_bias.view(1, -1, 1, 1)

    return stream_outputs, integrated_out
```

**Helper method (for analysis and debugging):**
```python
def forward_stream(self, stream_idx: int, stream_input: Tensor) -> Tensor:
    """Forward single stream through its dedicated pathway (for analysis/debugging)."""
    return F.conv2d(
        stream_input,
        self.stream_weights[stream_idx],
        self.stream_biases[stream_idx] if self.stream_biases else None,
        self.stride, self.padding, self.dilation, self.groups
    )
```

#### 1.2 LIBatchNorm2d Class

**Current signature:**
```python
def __init__(self,
             stream1_num_features: int,
             stream2_num_features: int,
             integrated_num_features: int,
             eps: float = 1e-5,
             momentum: float = 0.1,
             affine: bool = True,
             track_running_stats: bool = True,
             device=None,
             dtype=None)
```

**Refactored signature:**
```python
def __init__(self,
             stream_num_features: List[int],    # NEW: [64, 64, 64]
             integrated_num_features: int,      # UNCHANGED
             eps: float = 1e-5,
             momentum: float = 0.1,
             affine: bool = True,
             track_running_stats: bool = True,
             device=None,
             dtype=None)
```

**Parameter changes:**
```python
# NEW
self.num_streams = len(stream_num_features)
self.stream_num_features = stream_num_features

# Affine parameters (weights and biases) per stream
if affine:
    self.stream_weights = nn.ParameterList([
        Parameter(torch.empty(num_features, **factory_kwargs))
        for num_features in stream_num_features
    ])
    self.stream_biases = nn.ParameterList([
        Parameter(torch.empty(num_features, **factory_kwargs))
        for num_features in stream_num_features
    ])
else:
    self.stream_weights = None
    self.stream_biases = None

# Running statistics per stream
if track_running_stats:
    self.stream_running_means = [
        self.register_buffer(f'stream_{i}_running_mean',
                           torch.zeros(num_features, **factory_kwargs))
        for i, num_features in enumerate(stream_num_features)
    ]
    self.stream_running_vars = [
        self.register_buffer(f'stream_{i}_running_var',
                           torch.ones(num_features, **factory_kwargs))
        for i, num_features in enumerate(stream_num_features)
    ]
else:
    self.stream_running_means = None
    self.stream_running_vars = None

# Integrated stream params (unchanged structure)
if affine:
    self.integrated_weight = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
    self.integrated_bias = Parameter(torch.empty(integrated_num_features, **factory_kwargs))
if track_running_stats:
    self.register_buffer('integrated_running_mean', torch.zeros(integrated_num_features, **factory_kwargs))
    self.register_buffer('integrated_running_var', torch.ones(integrated_num_features, **factory_kwargs))
    # ⚠️ CORRECTED: num_batches_tracked is SHARED across all streams (single counter)
    self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, **factory_kwargs))
```

**Forward method:**
```python
def forward(self,
            stream_inputs: List[Tensor],
            integrated_input: Optional[Tensor] = None
            ) -> Tuple[List[Tensor], Tensor]:

    # Process each stream
    stream_outputs = []
    for i in range(self.num_streams):
        stream_out = F.batch_norm(
            stream_inputs[i],
            self.stream_running_means[i] if self.track_running_stats else None,
            self.stream_running_vars[i] if self.track_running_stats else None,
            self.stream_weights[i] if self.affine else None,
            self.stream_biases[i] if self.affine else None,
            self.training,
            self.momentum,
            self.eps
        )
        stream_outputs.append(stream_out)

    # Process integrated stream
    if integrated_input is not None:
        integrated_out = F.batch_norm(
            integrated_input,
            self.integrated_running_mean if self.track_running_stats else None,
            self.integrated_running_var if self.track_running_stats else None,
            self.integrated_weight if self.affine else None,
            self.integrated_bias if self.affine else None,
            self.training,
            self.momentum,
            self.eps
        )
    else:
        integrated_out = None

    # Update batch counter
    if self.track_running_stats and self.training:
        self.num_batches_tracked += 1

    return stream_outputs, integrated_out
```

---

### 2. pooling.py - Pooling Layers

#### 2.1 LIMaxPool2d

**Current forward:**
```python
def forward(self,
            stream1_input: Tensor,
            stream2_input: Tensor,
            integrated_input: Optional[Tensor] = None
            ) -> tuple[Tensor, Tensor, Tensor]:

    stream1_out = F.max_pool2d(stream1_input, self.kernel_size, self.stride, ...)
    stream2_out = F.max_pool2d(stream2_input, self.kernel_size, self.stride, ...)
    integrated_out = F.max_pool2d(integrated_input, ...) if integrated_input is not None else None

    return stream1_out, stream2_out, integrated_out
```

**Refactored forward:**
```python
def forward(self,
            stream_inputs: List[Tensor],
            integrated_input: Optional[Tensor] = None
            ) -> Tuple[List[Tensor], Tensor]:

    # Pool each stream
    stream_outputs = [
        F.max_pool2d(stream_input, self.kernel_size, self.stride,
                    self.padding, self.dilation, self.ceil_mode, self.return_indices)
        for stream_input in stream_inputs
    ]

    # Pool integrated stream
    integrated_out = F.max_pool2d(
        integrated_input, self.kernel_size, self.stride,
        self.padding, self.dilation, self.ceil_mode, self.return_indices
    ) if integrated_input is not None else None

    return stream_outputs, integrated_out
```

#### 2.2 LIAdaptiveAvgPool2d

**Same pattern as MaxPool2d** - replace dual stream processing with list processing.

---

### 3. container.py - Activation and Sequential

#### 3.1 LIReLU

**Current forward:**
```python
def forward(self,
            stream1_input: Tensor,
            stream2_input: Tensor,
            integrated_input: Optional[Tensor] = None
            ) -> tuple[Tensor, Tensor, Tensor]:

    stream1_out = F.relu(stream1_input, inplace=self.inplace)
    stream2_out = F.relu(stream2_input, inplace=self.inplace)
    integrated_out = F.relu(integrated_input, inplace=self.inplace) if integrated_input is not None else None

    return stream1_out, stream2_out, integrated_out
```

**Refactored forward:**
```python
def forward(self,
            stream_inputs: List[Tensor],
            integrated_input: Optional[Tensor] = None
            ) -> Tuple[List[Tensor], Tensor]:

    stream_outputs = [
        F.relu(stream_input, inplace=self.inplace)
        for stream_input in stream_inputs
    ]

    integrated_out = F.relu(integrated_input, inplace=self.inplace) if integrated_input is not None else None

    return stream_outputs, integrated_out
```

#### 3.2 LISequential

**Current implementation** (already generic):
```python
def forward(self, *inputs):
    for module in self:
        inputs = module(*inputs)
    return inputs
```

**Should work as-is**, but verify that unpacking works with `(List[Tensor], Tensor)` tuples.

---

### 4. blocks.py - ResNet Building Blocks

#### 4.1 Helper Functions

**Current:**
```python
def li_conv3x3(stream1_in_planes: int,
               stream2_in_planes: int,
               integrated_in_planes: int,
               stream1_out_planes: int,
               stream2_out_planes: int,
               integrated_out_planes: int,
               stride: int = 1,
               groups: int = 1,
               dilation: int = 1) -> LIConv2d:
    return LIConv2d(
        stream1_in_planes, stream2_in_planes, integrated_in_planes,
        stream1_out_planes, stream2_out_planes, integrated_out_planes,
        kernel_size=3, stride=stride, padding=dilation,
        groups=groups, bias=False, dilation=dilation
    )
```

**Refactored:**
```python
def li_conv3x3(stream_in_planes: List[int],
               integrated_in_planes: int,
               stream_out_planes: List[int],
               integrated_out_planes: int,
               stride: int = 1,
               groups: int = 1,
               dilation: int = 1) -> LIConv2d:
    return LIConv2d(
        stream_in_planes,
        stream_out_planes,
        integrated_in_planes,
        integrated_out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )
```

**Same for `li_conv1x1()`**

#### 4.2 LIBasicBlock

**Current signature:**
```python
def __init__(self,
             stream1_inplanes: int,
             stream2_inplanes: int,
             integrated_inplanes: int,
             stream1_planes: int,
             stream2_planes: int,
             stride: int = 1,
             downsample: Optional[nn.Module] = None,
             groups: int = 1,
             base_width: int = 64,
             dilation: int = 1,
             norm_layer: Optional[Callable[..., nn.Module]] = None)
```

**Refactored signature:**
```python
def __init__(self,
             stream_inplanes: List[int],        # NEW: [64, 64, 64]
             integrated_inplanes: int,          # UNCHANGED
             stream_planes: List[int],          # NEW: [64, 64, 64]
             stride: int = 1,
             downsample: Optional[nn.Module] = None,
             groups: int = 1,
             base_width: int = 64,
             dilation: int = 1,
             norm_layer: Optional[Callable[..., nn.Module]] = None)
```

**Changes:**
```python
# Store stream info
self.num_streams = len(stream_inplanes)
self.stream_inplanes = stream_inplanes
self.stream_planes = stream_planes

# Compute integrated planes (typically same as stream planes)
integrated_planes = stream_planes[0]  # Assuming all streams have same output channels

# Build layers
self.conv1 = li_conv3x3(stream_inplanes, integrated_inplanes,
                       stream_planes, integrated_planes, stride)
self.bn1 = norm_layer(stream_planes, integrated_planes)
self.relu = LIReLU(inplace=True)
self.conv2 = li_conv3x3(stream_planes, integrated_planes,
                       stream_planes, integrated_planes)
self.bn2 = norm_layer(stream_planes, integrated_planes)

# Downsample if needed
self.downsample = downsample
self.stride = stride
```

**Forward changes:**
```python
# OLD
def forward(self,
            stream1_input: Tensor,
            stream2_input: Tensor,
            integrated_input: Optional[Tensor] = None
            ) -> tuple[Tensor, Tensor, Tensor]:

    # Save identity for residual
    identity_s1 = stream1_input
    identity_s2 = stream2_input
    identity_ic = integrated_input

    # Forward through layers
    s1, s2, ic = self.conv1(stream1_input, stream2_input, integrated_input)
    s1, s2, ic = self.bn1(s1, s2, ic)
    s1, s2, ic = self.relu(s1, s2, ic)
    s1, s2, ic = self.conv2(s1, s2, ic)
    s1, s2, ic = self.bn2(s1, s2, ic)

    # Downsample if needed
    if self.downsample is not None:
        identity_s1, identity_s2, identity_ic = self.downsample(stream1_input, stream2_input, integrated_input)

    # Residual connections
    s1 += identity_s1
    s2 += identity_s2
    if identity_ic is not None:
        ic += identity_ic

    # Final activation
    s1, s2, ic = self.relu(s1, s2, ic)

    return s1, s2, ic

# NEW
def forward(self,
            stream_inputs: List[Tensor],
            integrated_input: Optional[Tensor] = None
            ) -> Tuple[List[Tensor], Tensor]:

    # Save identity for residual (per stream)
    identity_streams = [stream.clone() for stream in stream_inputs]
    identity_integrated = integrated_input.clone() if integrated_input is not None else None

    # Forward through layers
    stream_outputs, integrated_out = self.conv1(stream_inputs, integrated_input)
    stream_outputs, integrated_out = self.bn1(stream_outputs, integrated_out)
    stream_outputs, integrated_out = self.relu(stream_outputs, integrated_out)
    stream_outputs, integrated_out = self.conv2(stream_outputs, integrated_out)
    stream_outputs, integrated_out = self.bn2(stream_outputs, integrated_out)

    # Downsample if needed
    if self.downsample is not None:
        identity_streams, identity_integrated = self.downsample(stream_inputs, integrated_input)

    # Residual connections (per stream)
    for i in range(self.num_streams):
        stream_outputs[i] = stream_outputs[i] + identity_streams[i]

    if identity_integrated is not None:
        integrated_out = integrated_out + identity_integrated

    # Final activation
    stream_outputs, integrated_out = self.relu(stream_outputs, integrated_out)

    return stream_outputs, integrated_out
```

#### 4.3 LIBottleneck

**Same pattern as BasicBlock** - replace dual stream parameters with lists.

---

### 5. li_net.py - Main Network

#### 5.1 LINet.__init__()

**Current signature:**
```python
def __init__(self,
             block: type[Union[LIBasicBlock, LIBottleneck]],
             layers: list[int],
             num_classes: int,
             stream1_input_channels: int = 3,
             stream2_input_channels: int = 1,
             groups: int = 1,
             width_per_group: int = 64,
             replace_stride_with_dilation: Optional[list[bool]] = None,
             norm_layer: Optional[Callable[..., nn.Module]] = None,
             dropout_p: float = 0.0,
             **kwargs)
```

**Refactored signature:**
```python
def __init__(self,
             block: type[Union[LIBasicBlock, LIBottleneck]],
             layers: list[int],
             num_classes: int,
             stream_input_channels: List[int] = [3, 1, 1],  # NEW: RGB, Depth, Orth
             groups: int = 1,
             width_per_group: int = 64,
             replace_stride_with_dilation: Optional[list[bool]] = None,
             norm_layer: Optional[Callable[..., nn.Module]] = None,
             dropout_p: float = 0.0,
             **kwargs)
```

**Changes:**
```python
# Store stream info
self.num_streams = len(stream_input_channels)
self.stream_input_channels = stream_input_channels

# Initialize channel tracking for each stream
self.stream_inplanes = [64] * self.num_streams  # All start at 64 after conv1
self.integrated_inplanes = 64

# Build stem
self.conv1 = LIConv2d(
    stream_input_channels,      # [3, 1, 1]
    [64] * self.num_streams,    # [64, 64, 64]
    0,                          # No previous integrated
    64,                         # Integrated output channels
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
self.bn1 = norm_layer([64] * self.num_streams, 64)
self.relu = LIReLU(inplace=True)
self.maxpool = LIMaxPool2d(kernel_size=3, stride=2, padding=1)

# Build ResNet layers
self.layer1 = self._make_layer(block, [64] * self.num_streams, layers[0])
self.layer2 = self._make_layer(block, [128] * self.num_streams, layers[1], stride=2, dilate=False)
self.layer3 = self._make_layer(block, [256] * self.num_streams, layers[2], stride=2, dilate=False)
self.layer4 = self._make_layer(block, [512] * self.num_streams, layers[3], stride=2, dilate=False)

# Final classifier (uses integrated stream only)
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.dropout = nn.Dropout(p=dropout_p)
self.fc = nn.Linear(512 * block.expansion, num_classes)

# Auxiliary stream classifiers (optional for monitoring)
self.fc_streams = nn.ModuleList([
    nn.Linear(512 * block.expansion, num_classes)
    for _ in range(self.num_streams)
])
```

#### 5.2 LINet._make_layer()

**Current:**
```python
def _make_layer(self,
                block: type[Union[LIBasicBlock, LIBottleneck]],
                stream1_planes: int,
                stream2_planes: int,
                blocks: int,
                stride: int = 1,
                dilate: bool = False) -> LISequential:
    # Build layer with channel tracking
    # Updates self.stream1_inplanes, self.stream2_inplanes, self.integrated_inplanes
    ...
```

**Refactored:**
```python
def _make_layer(self,
                block: type[Union[LIBasicBlock, LIBottleneck]],
                stream_planes: List[int],        # NEW: [128, 128, 128]
                blocks: int,
                stride: int = 1,
                dilate: bool = False) -> LISequential:

    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation

    if dilate:
        self.dilation *= stride
        stride = 1

    # Downsample if stride != 1 or channels change
    if stride != 1 or self.stream_inplanes[0] != stream_planes[0] * block.expansion:
        downsample = LISequential(
            li_conv1x1(
                self.stream_inplanes,
                self.integrated_inplanes,
                [p * block.expansion for p in stream_planes],
                stream_planes[0] * block.expansion,
                stride
            ),
            norm_layer(
                [p * block.expansion for p in stream_planes],
                stream_planes[0] * block.expansion
            ),
        )

    # Build blocks
    layers = []

    # First block (may have stride and downsample)
    layers.append(
        block(
            self.stream_inplanes,
            self.integrated_inplanes,
            stream_planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer
        )
    )

    # Update channel counts
    self.stream_inplanes = [p * block.expansion for p in stream_planes]
    self.integrated_inplanes = stream_planes[0] * block.expansion

    # Remaining blocks
    for _ in range(1, blocks):
        layers.append(
            block(
                self.stream_inplanes,
                self.integrated_inplanes,
                stream_planes,
                stride=1,
                downsample=None,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer
            )
        )

    return LISequential(*layers)
```

#### 5.3 LINet.forward()

**Current:**
```python
def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> Tensor:
    # Stem
    s1, s2, ic = self.conv1(stream1_input, stream2_input, None)
    s1, s2, ic = self.bn1(s1, s2, ic)
    s1, s2, ic = self.relu(s1, s2, ic)
    s1, s2, ic = self.maxpool(s1, s2, ic)

    # ResNet layers
    s1, s2, ic = self.layer1(s1, s2, ic)
    s1, s2, ic = self.layer2(s1, s2, ic)
    s1, s2, ic = self.layer3(s1, s2, ic)
    s1, s2, ic = self.layer4(s1, s2, ic)

    # Classification (integrated stream only)
    integrated_pooled = self.avgpool(ic)
    features = torch.flatten(integrated_pooled, 1)
    features = self.dropout(features)
    logits = self.fc(features)

    return logits
```

**Refactored (N-stream, clean break):**
```python
def forward(self, stream_inputs: List[Tensor]) -> Tensor:
    """
    Forward pass through the model.

    Args:
        stream_inputs: List of N stream tensors, e.g., [rgb, depth, orth]

    Returns:
        logits: Classification logits from integrated stream
    """
    assert len(stream_inputs) == self.num_streams, \
        f"Expected {self.num_streams} streams, got {len(stream_inputs)}"

    # Stem
    stream_outputs, integrated = self.conv1(stream_inputs, None)
    stream_outputs, integrated = self.bn1(stream_outputs, integrated)
    stream_outputs, integrated = self.relu(stream_outputs, integrated)
    stream_outputs, integrated = self.maxpool(stream_outputs, integrated)

    # ResNet layers
    stream_outputs, integrated = self.layer1(stream_outputs, integrated)
    stream_outputs, integrated = self.layer2(stream_outputs, integrated)
    stream_outputs, integrated = self.layer3(stream_outputs, integrated)
    stream_outputs, integrated = self.layer4(stream_outputs, integrated)

    # Classification (integrated stream only)
    integrated_pooled = self.avgpool(integrated)
    features = torch.flatten(integrated_pooled, 1)
    features = self.dropout(features)
    logits = self.fc(features)

    return logits
```

#### 5.4 Auxiliary Classifiers & Stream Pathways

**Purpose**: Auxiliary classifiers enable per-stream monitoring and stream-specific early stopping.

**Refactored implementation:**
```python
# In __init__():
# Auxiliary classifiers (one per stream) - used for monitoring and early stopping
self.fc_streams = nn.ModuleList([
    nn.Linear(512 * block.expansion, num_classes)
    for _ in range(self.num_streams)
])

# Stream pathway method (for monitoring and analysis)
def _forward_stream_pathway(self, stream_idx: int, stream_input: Tensor) -> Tensor:
    """
    Forward a single stream through its dedicated pathway.
    Used for stream-specific monitoring and analysis.

    Args:
        stream_idx: Index of the stream to forward (0=RGB, 1=Depth, 2=Orth, etc.)
        stream_input: Input tensor for this stream

    Returns:
        Stream-specific features after avgpool (before FC)
    """
    # This requires forwarding through all layers for just one stream
    # Implementation matches current _forward_stream1_pathway pattern
    # Returns features ready for auxiliary classifier
    pass
```

#### 5.5 Stream-Specific Early Stopping

**Overview**: Allows individual streams to stop training early while others continue.

**Refactored implementation:**
```python
def _freeze_stream(self, stream_idx: int, restore_best_weights: bool = True):
    """
    Freeze training for a specific stream.

    Args:
        stream_idx: Which stream to freeze (0, 1, 2, ...)
        restore_best_weights: Whether to restore best weights before freezing
    """
    # Restore best weights for this stream if requested
    if restore_best_weights and hasattr(self, f'_best_stream_{stream_idx}_weights'):
        best_weights = getattr(self, f'_best_stream_{stream_idx}_weights')
        for name, param in self.named_parameters():
            if f'stream_weights.{stream_idx}' in name or f'stream_biases.{stream_idx}' in name:
                if name in best_weights:
                    param.data.copy_(best_weights[name])

    # Freeze stream parameters (but keep integration weights trainable!)
    for name, param in self.named_parameters():
        if f'stream_weights.{stream_idx}' in name or f'stream_biases.{stream_idx}' in name:
            param.requires_grad = False

def _check_stream_early_stopping(self, stream_idx: int, stream_metric: float,
                                  patience: int, min_delta: float, monitor: str) -> bool:
    """
    Check if stream should stop training early.

    Returns:
        True if stream should be frozen, False otherwise
    """
    # Implementation tracks per-stream patience counters and best metrics
    # Returns True when patience exhausted
    pass
```

#### 5.6 fit() Method Signature

**Full signature with all parameters:**
```python
def fit(self,
        train_loader,
        val_loader=None,
        epochs=10,
        callbacks=None,
        verbose=True,
        save_path=None,
        early_stopping=False,
        patience=10,
        min_delta=0.001,
        monitor='val_loss',
        restore_best_weights=True,
        gradient_accumulation_steps=1,
        grad_clip_norm=None,
        clear_cache_per_epoch=False,
        stream_monitoring=False,
        stream_early_stopping=False,
        stream_patience: dict = None,  # e.g., {0: 10, 1: 15, 2: 20}
        stream_min_delta: dict = None  # e.g., {0: 0.001, 1: 0.001, 2: 0.001}
        ) -> dict:
    """
    Train the model with support for:
    - Gradient accumulation
    - Gradient clipping
    - Stream-specific monitoring
    - Stream-specific early stopping
    - AMP (if enabled in compile())
    - OneCycleLR and other scheduler types
    """
    # Refactor to handle N streams instead of hardcoded stream1/stream2
    # Use stream_patience dict with stream indices as keys
    # Use self.fc_streams[i] for auxiliary classifiers
    pass
```

---

### 6. gradient_monitor.py - Gradient Monitoring

**Changes needed:**
- Update stream detection logic from checking "stream1" and "stream2" in parameter names
- Change to iterate over `stream_weights` ParameterList indices

**Example:**
```python
# OLD
if 'stream1' in param_name:
    stream1_grad_norm = param.grad.norm().item()
elif 'stream2' in param_name:
    stream2_grad_norm = param.grad.norm().item()

# NEW
import re
stream_weight_match = re.match(r'.*stream_weights\.(\d+).*', param_name)
if stream_weight_match:
    stream_idx = int(stream_weight_match.group(1))
    stream_grad_norms[stream_idx] = param.grad.norm().item()
```

---

### 7. stream_monitor.py - Stream-Specific Monitoring

**Changes needed:**
- Replace hardcoded metrics for stream1/stream2 with dynamic lists
- Update logging to handle arbitrary N streams

**Example:**
```python
# OLD
stream1_metrics = {...}
stream2_metrics = {...}

# NEW
stream_metrics = [
    {...}  # Stream 0 (RGB)
    {...}  # Stream 1 (Depth)
    {...}  # Stream 2 (Orthogonal)
]
```

---

## 6. Factory Functions & Auxiliary Classifiers

### 6.1 Factory Functions - N-Stream Signature

**File**: `src/models/linear_integration/li_net3/li_net.py`

All factory functions (`li_resnet18()`, `li_resnet34()`, etc.) need signature updates.

**CURRENT (2-stream):**
```python
def li_resnet18(num_classes: int = 1000, **kwargs) -> LINet:
    """Create a Linear Integration ResNet-18 model."""
    return LINet(
        LIBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        **kwargs
    )
```

**Usage (CURRENT):**
```python
model = li_resnet18(
    num_classes=15,
    stream1_input_channels=3,  # RGB
    stream2_input_channels=1,  # Depth
    dropout_p=0.5
)
```

**TARGET (N-stream):**
```python
def li_resnet18(
    num_classes: int = 1000,
    stream_input_channels: Optional[List[int]] = None,
    **kwargs
) -> LINet:
    """
    Create a Linear Integration ResNet-18 model.

    Args:
        num_classes: Number of output classes
        stream_input_channels: List of input channels for each stream.
                              Default: [3, 1] for RGB + Depth (backward compatible)
                              For 3 streams: [3, 1, 1] for RGB + Depth + Orth
        **kwargs: Additional arguments passed to LINet constructor

    Returns:
        LINet model instance
    """
    # Default to 2-stream for backward compatibility with existing code
    if stream_input_channels is None:
        stream_input_channels = [3, 1]  # RGB + Depth

    return LINet(
        LIBasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        stream_input_channels=stream_input_channels,
        **kwargs
    )
```

**Usage (TARGET):**
```python
# 2-stream (backward compatible - default)
model = li_resnet18(num_classes=15, dropout_p=0.5)

# 3-stream (RGB + Depth + Orth)
model = li_resnet18(
    num_classes=15,
    stream_input_channels=[3, 1, 1],  # RGB, Depth, Orth
    dropout_p=0.5
)

# 4-stream (future extensibility)
model = li_resnet18(
    num_classes=15,
    stream_input_channels=[3, 1, 1, 1],  # RGB, Depth, Orth, Surface Normals
    dropout_p=0.5
)
```

**Apply same changes to all factory functions:**
- `li_resnet18()` ✓
- `li_resnet34()` ✓
- `li_resnet50()` ✓
- `li_resnet101()` ✓
- `li_resnet152()` ✓

---

### 6.2 Auxiliary Classifiers - ModuleList Refactoring

**Current Implementation:**
```python
# In LINet.__init__() - lines 155-156
self.fc_stream1 = nn.Linear(feature_dim, self.num_classes)
self.fc_stream2 = nn.Linear(feature_dim, self.num_classes)
```

**TARGET (N-stream):**
```python
# In LINet.__init__()
# Auxiliary classifiers (one per stream) - used for monitoring and early stopping
self.fc_streams = nn.ModuleList([
    nn.Linear(feature_dim, self.num_classes)
    for _ in range(self.num_streams)
])
```

**Update all usages:**

1. **Lines 716-717** (compile method):
```python
# CURRENT:
self.fc_stream1.weight, self.fc_stream1.bias,
self.fc_stream2.weight, self.fc_stream2.bias

# TARGET:
*[param for fc in self.fc_streams for param in [fc.weight, fc.bias]]
```

2. **Lines 1276-1277, 1309-1310, 1429-1430** (fit method - stream monitoring):
```python
# CURRENT:
stream1_outputs = self.fc_stream1(stream1_features_detached)
stream2_outputs = self.fc_stream2(stream2_features_detached)

# TARGET (loop over all streams):
for stream_idx in range(self.num_streams):
    stream_features = self._forward_stream_pathway(stream_idx, stream_inputs[stream_idx])
    stream_outputs = self.fc_streams[stream_idx](stream_features.detach())
    # Calculate stream-specific metrics...
```

3. **Lines 1714, 1724** (evaluate method):
```python
# CURRENT:
stream1_outputs = self.fc_stream1(stream1_features)  # Auxiliary classifier!
stream2_outputs = self.fc_stream2(stream2_features)  # Auxiliary classifier!

# TARGET:
stream_outputs_list = []
for stream_idx in range(self.num_streams):
    stream_features = self._forward_stream_pathway(stream_idx, stream_inputs[stream_idx])
    stream_outputs = self.fc_streams[stream_idx](stream_features)
    stream_outputs_list.append(stream_outputs)
```

---

### 6.3 Stream Pathway Methods - Refactoring

**Current Implementation:**
```python
# Lines 249-272 in abstract_model.py - TWO separate methods
def _forward_stream1_pathway(self, stream1_input: Tensor) -> Tensor:
    """Forward stream1 through its dedicated pathway."""
    # Implementation for stream1
    pass

def _forward_stream2_pathway(self, stream2_input: Tensor) -> Tensor:
    """Forward stream2 through its dedicated pathway."""
    # Implementation for stream2
    pass
```

**TARGET (N-stream, single method):**
```python
def _forward_stream_pathway(self, stream_idx: int, stream_input: Tensor) -> Tensor:
    """
    Forward a single stream through its dedicated pathway.
    Used for stream-specific monitoring and analysis.

    Args:
        stream_idx: Index of the stream to forward (0=RGB, 1=Depth, 2=Orth, etc.)
        stream_input: Input tensor for this stream

    Returns:
        Stream-specific features after avgpool (before FC classifier)
    """
    # Forward through network with only one stream active
    # Initialize all streams to zeros except the target stream
    stream_inputs = [torch.zeros_like(stream_input) for _ in range(self.num_streams)]
    stream_inputs[stream_idx] = stream_input

    # Forward through all layers (only stream_idx has real data)
    # Note: We ignore integrated output (_) since we only want the specific stream's pathway
    stream_outputs, integrated = self.conv1(stream_inputs, None)
    stream_outputs, integrated = self.bn1(stream_outputs, integrated)
    stream_outputs, integrated = self.relu(stream_outputs, integrated)
    stream_outputs, integrated = self.maxpool(stream_outputs, integrated)

    stream_outputs, integrated = self.layer1(stream_outputs, integrated)
    stream_outputs, integrated = self.layer2(stream_outputs, integrated)
    stream_outputs, integrated = self.layer3(stream_outputs, integrated)
    stream_outputs, integrated = self.layer4(stream_outputs, integrated)

    # Extract target stream output
    stream_output = stream_outputs[stream_idx]

    # Pool and flatten
    pooled = self.avgpool(stream_output)
    features = torch.flatten(pooled, 1)

    return features
```

**Usage:**
```python
# OLD (hardcoded for 2 streams):
stream1_features = model._forward_stream1_pathway(rgb)
stream2_features = model._forward_stream2_pathway(depth)

# NEW (dynamic for N streams):
stream_features_list = []
for stream_idx in range(model.num_streams):
    features = model._forward_stream_pathway(stream_idx, stream_inputs[stream_idx])
    stream_features_list.append(features)
```

---

## Testing Strategy

### 1. Unit Tests per Component

```python
def test_liconv2d_n_streams():
    """Test LIConv2d with N=3 streams."""
    conv = LIConv2d(
        stream_in_channels=[3, 1, 1],
        stream_out_channels=[64, 64, 64],
        integrated_in_channels=0,
        integrated_out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3
    )

    # Create dummy inputs
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    orth = torch.randn(2, 1, 224, 224)

    # Forward pass
    stream_outputs, integrated = conv([rgb, depth, orth], None)

    # Check outputs
    assert len(stream_outputs) == 3
    assert stream_outputs[0].shape == (2, 64, 112, 112)
    assert stream_outputs[1].shape == (2, 64, 112, 112)
    assert stream_outputs[2].shape == (2, 64, 112, 112)
    assert integrated.shape == (2, 64, 112, 112)
```

### 2. Integration Tests

```python
def test_linet_n3_forward():
    """Test full LINet forward pass with N=3 streams."""
    model = li_resnet18(
        num_classes=15,
        stream_input_channels=[3, 1, 1]  # RGB, Depth, Orth
    )

    # Create dummy inputs
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    orth = torch.randn(2, 1, 224, 224)

    # Forward pass
    logits = model([rgb, depth, orth])

    # Check output
    assert logits.shape == (2, 15)
```

### 3. Backward Compatibility Test

```python
def test_n2_matches_original():
    """Verify N=2 refactored version matches original 2-stream behavior."""

    # Original model
    model_old = LINet_Original(...)

    # Refactored model with N=2
    model_new = LINet(
        stream_input_channels=[3, 1]  # Only RGB and Depth
    )

    # Copy weights (if compatible)
    # ... weight migration code ...

    # Compare outputs
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)

    logits_old = model_old(rgb, depth)
    logits_new = model_new([rgb, depth])

    assert torch.allclose(logits_old, logits_new, atol=1e-6)
```

### 4. Gradient Flow Test

```python
def test_gradient_flow_n_streams():
    """Ensure gradients flow to all streams."""
    model = li_resnet18(stream_input_channels=[3, 1, 1])

    # Forward + backward
    rgb = torch.randn(2, 3, 224, 224, requires_grad=True)
    depth = torch.randn(2, 1, 224, 224, requires_grad=True)
    orth = torch.randn(2, 1, 224, 224, requires_grad=True)

    logits = model([rgb, depth, orth])
    loss = logits.sum()
    loss.backward()

    # Check gradients exist
    assert rgb.grad is not None
    assert depth.grad is not None
    assert orth.grad is not None

    # Check stream weights have gradients
    for i, stream_weight in enumerate(model.conv1.stream_weights):
        assert stream_weight.grad is not None, f"Stream {i} has no gradient"
```

---

## Migration Strategy

### Option A: Clean Break (Recommended)
- Implement N-stream from scratch
- No backward compatibility with 2-stream checkpoints
- Cleaner, more maintainable code

### Option B: Checkpoint Migration Script
- Implement converter to migrate old 2-stream weights to new N-stream format
- Allows loading pretrained 2-stream models
- More complex implementation

**Recommendation:** Start with **Option A**, add migration script later if needed.

---

## Implementation Checklist

### Phase 1: Core Layers
- [ ] Refactor `LIConv2d` to use `nn.ParameterList` for stream weights
- [ ] Refactor `LIBatchNorm2d` to use `nn.ParameterList` for stream parameters
- [ ] Update `LIReLU` for list-based forward
- [ ] Update `LIMaxPool2d` for list-based forward
- [ ] Update `LIAdaptiveAvgPool2d` for list-based forward
- [ ] Unit tests for each layer

### Phase 2: Building Blocks
- [ ] Refactor `li_conv3x3()` helper
- [ ] Refactor `li_conv1x1()` helper
- [ ] Refactor `LIBasicBlock`
- [ ] Refactor `LIBottleneck`
- [ ] Unit tests for blocks

### Phase 3: Main Network
- [ ] Refactor `LINet.__init__()`
- [ ] Refactor `LINet._make_layer()`
- [ ] Refactor `LINet.forward()`
- [ ] Update auxiliary classifiers to `nn.ModuleList`
- [ ] Integration test for full network

### Phase 4: Factory Functions & Auxiliary Classifiers
- [ ] Update `li_resnet18()` signature for N-stream
- [ ] Update `li_resnet34()` signature for N-stream
- [ ] Update `li_resnet50()` signature for N-stream
- [ ] Update `li_resnet101()` signature for N-stream
- [ ] Update `li_resnet152()` signature for N-stream
- [ ] Update auxiliary classifiers (`fc_stream1`, `fc_stream2`) to `nn.ModuleList` (`fc_streams`)
- [ ] Update all uses of `fc_stream1` and `fc_stream2` to use `fc_streams[i]`

### Phase 5: Utilities
- [ ] Update `GradientMonitor` for N streams
- [ ] Update `StreamMonitor` for N streams (if needed - may already be generic)
- [ ] Refactor early stopping methods for N streams

### Phase 6: Testing & Validation
- [ ] Test N=1 (single stream - should work like ResNet)
- [ ] Test N=2 (should match original behavior if possible)
- [ ] Test N=3 (RGB + Depth + Orthogonal)
- [ ] Test N=4+ (future extensibility)
- [ ] Gradient flow tests
- [ ] Training smoke test on small dataset

---

## Expected Parameter Increase

### ResNet-18 Example

**Current (N=2):**
- ~11.7M total parameters
- Stream-specific: ~5.8M
- Integrated: ~5.9M

**Refactored (N=3):**
- ~15.3M total parameters (+31%)
- Stream-specific: ~8.7M (3 streams)
- Integrated: ~6.6M (3 integration weights per layer)

**Breakdown per layer type:**

| Layer | N=2 Params | N=3 Params | Increase |
|-------|-----------|-----------|----------|
| conv1 (7×7) | 20,736 | 27,968 | +35% |
| layer1 (64 ch) | ~147K | ~221K | +50% |
| layer2 (128 ch) | ~590K | ~885K | +50% |
| layer3 (256 ch) | ~2.36M | ~3.54M | +50% |
| layer4 (512 ch) | ~9.44M | ~14.16M | +50% |

---

## 7. Support Files - N-Stream Updates (Clean Break)

The following support files outside the li_net3 module require updates to support the new N-stream architecture. Following the **Clean Break** strategy, these will be refactored to only support the N-stream API. Legacy 2-stream code will be updated separately to use the new API.

### 7.1 abstract_model.py - Base Model Interface

**File**: `src/models/abstracts/abstract_model.py`

**Current Issues**:
1. **Lines 65-66**: Hardcoded `self.stream1_inplanes = 64` and `self.stream2_inplanes = 64`
2. **Lines 128-208**: `get_stream_parameter_groups()` uses positional args for stream1/stream2
3. **Lines 235-246**: `forward()` signature takes exactly 2 stream inputs
4. **Lines 249-272**: `_forward_stream1_pathway()` and `_forward_stream2_pathway()` hardcoded

**Required Changes**:

#### 7.1.1 Stream Inplanes Initialization

**CURRENT:**
```python
# Lines 65-66
self.stream1_inplanes = 64
self.stream2_inplanes = 64
```

**TARGET:**
```python
# Dynamic initialization based on num_streams
self.num_streams = len(stream_input_channels)
self.stream_inplanes = [64] * self.num_streams
```

#### 7.1.2 Parameter Groups Method - N-Stream Only

**CURRENT (2-stream only):**
```python
def get_stream_parameter_groups(
    self,
    stream1_lr: Optional[float] = None,
    stream2_lr: Optional[float] = None,
    shared_lr: Optional[float] = None,
    stream1_weight_decay: float = 0.0,
    stream2_weight_decay: float = 0.0,
    shared_weight_decay: float = 0.0
) -> List[Dict[str, Any]]:
    """Returns parameter groups with stream-specific learning rates."""
    # Hardcoded for stream1 and stream2
    param_groups = []

    # Stream 1 parameters
    stream1_params = [p for name, p in self.named_parameters() if '.stream1_' in name]
    if stream1_params:
        param_groups.append({
            'params': stream1_params,
            'lr': stream1_lr,
            'weight_decay': stream1_weight_decay
        })

    # Stream 2 parameters
    stream2_params = [p for name, p in self.named_parameters() if '.stream2_' in name]
    if stream2_params:
        param_groups.append({
            'params': stream2_params,
            'lr': stream2_lr,
            'weight_decay': stream2_weight_decay
        })

    # Shared/integrated parameters
    shared_params = [p for name, p in self.named_parameters()
                     if '.stream1_' not in name and '.stream2_' not in name]
    if shared_params:
        param_groups.append({
            'params': shared_params,
            'lr': shared_lr,
            'weight_decay': shared_weight_decay
        })

    return param_groups
```

**TARGET (N-stream, clean break):**
```python
def get_stream_parameter_groups(
    self,
    stream_lrs: Dict[str, float],
    stream_weight_decays: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Returns parameter groups with stream-specific learning rates.

    Usage:
        get_stream_parameter_groups(
            stream_lrs={'stream1': 0.001, 'stream2': 0.0005, 'stream3': 0.0003, 'shared': 0.001},
            stream_weight_decays={'stream1': 0.0001, 'stream2': 0.0, 'stream3': 0.0, 'shared': 0.0001}
        )

    Args:
        stream_lrs: Dict mapping stream names to learning rates
                   Keys: 'stream1', 'stream2', 'stream3', ..., 'shared'
        stream_weight_decays: Dict mapping stream names to weight decay values
                             If None, defaults to 0.0 for all groups

    Returns:
        List of parameter group dicts suitable for PyTorch optimizers
    """
    import re

    if stream_weight_decays is None:
        stream_weight_decays = {k: 0.0 for k in stream_lrs.keys()}

    param_groups = []

    # Detect number of streams by examining parameter names
    stream_indices = set()
    for name in self.state_dict().keys():
        # Match patterns like .stream_weights.0., .stream_biases.2., etc.
        match = re.search(r'\.stream_(?:weights|biases|running_means|running_vars)\.(\d+)', name)
        if match:
            stream_indices.add(int(match.group(1)))

    num_streams = len(stream_indices)

    # Create parameter groups for each stream
    for i in range(num_streams):
        stream_key = f'stream{i+1}'  # stream1, stream2, stream3, etc.

        if stream_key not in stream_lrs:
            continue

        # Collect all parameters for this stream
        # Matches: .stream_weights.{i}., .stream_biases.{i}., etc.
        stream_params = [
            p for name, p in self.named_parameters()
            if re.search(rf'\.stream_(?:weights|biases)\.{i}(?:\.|$)', name)
        ]

        if stream_params:
            param_groups.append({
                'params': stream_params,
                'lr': stream_lrs[stream_key],
                'weight_decay': stream_weight_decays.get(stream_key, 0.0)
            })

    # Shared/integrated parameters (everything not stream-specific)
    if 'shared' in stream_lrs:
        shared_params = [
            p for name, p in self.named_parameters()
            if not re.search(r'\.stream_(?:weights|biases)\.(\d+)', name)
        ]

        if shared_params:
            param_groups.append({
                'params': shared_params,
                'lr': stream_lrs['shared'],
                'weight_decay': stream_weight_decays.get('shared', 0.0)
            })

    return param_groups
```

#### 7.1.3 Forward Method Signature

**CURRENT:**
```python
def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> Tensor:
    """Forward pass through the model."""
    raise NotImplementedError("Subclasses must implement forward()")
```

**TARGET (N-stream, clean break):**
```python
def forward(self, stream_inputs: List[Tensor]) -> Tensor:
    """
    Forward pass through the model.

    Args:
        stream_inputs: List of N stream tensors [rgb, depth, orth, ...]

    Returns:
        Classification logits
    """
    raise NotImplementedError("Subclasses must implement forward()")
```

#### 7.1.4 Stream Pathway Methods

**CURRENT:**
```python
def _forward_stream1_pathway(self, stream1_input: Tensor) -> Tensor:
    """Forward stream 1 through its dedicated pathway."""
    raise NotImplementedError("Subclasses must implement stream pathways")

def _forward_stream2_pathway(self, stream2_input: Tensor) -> Tensor:
    """Forward stream 2 through its dedicated pathway."""
    raise NotImplementedError("Subclasses must implement stream pathways")
```

**TARGET (N-stream, clean break):**
```python
def _forward_stream_pathway(self, stream_idx: int, stream_input: Tensor) -> Tensor:
    """
    Forward a single stream through its dedicated pathway.

    Args:
        stream_idx: Index of the stream (0, 1, 2, ...)
        stream_input: Input tensor for this stream

    Returns:
        Stream-specific features after avgpool (before FC layer)
    """
    raise NotImplementedError("Subclasses must implement stream pathways")
```

---

### 7.2 optimizers.py - Optimizer Setup

**File**: `src/training/optimizers.py`

**Current Issues**:
- `create_stream_optimizer()` function hardcoded for 2 streams (lines 13-111)
- Takes `stream1_lr`, `stream2_lr`, `shared_lr` as separate positional parameters

**Required Changes**:

#### 7.2.1 create_stream_optimizer() - N-Stream Only

**CURRENT (2-stream only):**
```python
def create_stream_optimizer(
    model,
    optimizer_type: str = 'adamw',
    stream1_lr: float = 0.001,
    stream2_lr: float = 0.001,
    shared_lr: float = 0.001,
    stream1_weight_decay: float = 0.0001,
    stream2_weight_decay: float = 0.0,
    shared_weight_decay: float = 0.0001,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with stream-specific learning rates.

    Only supports 2 streams (stream1 and stream2).
    """
    # Get parameter groups from model
    param_groups = model.get_stream_parameter_groups(
        stream1_lr=stream1_lr,
        stream2_lr=stream2_lr,
        shared_lr=shared_lr,
        stream1_weight_decay=stream1_weight_decay,
        stream2_weight_decay=stream2_weight_decay,
        shared_weight_decay=shared_weight_decay
    )

    # Create optimizer
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, betas=betas, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum, **kwargs)
    # ... other optimizer types

    return optimizer
```

**TARGET (N-stream, clean break):**
```python
def create_stream_optimizer(
    model,
    optimizer_type: str = 'adamw',
    stream_lrs: Dict[str, float],
    stream_weight_decays: Optional[Dict[str, float]] = None,
    momentum: float = 0.9,
    betas: tuple = (0.9, 0.999),
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with stream-specific learning rates.

    Usage:
        create_stream_optimizer(
            model,
            stream_lrs={'stream1': 0.001, 'stream2': 0.0005, 'stream3': 0.0003, 'shared': 0.001},
            stream_weight_decays={'stream1': 0.0001, 'stream2': 0.0, 'stream3': 0.0, 'shared': 0.0001}
        )

    Args:
        model: Model with get_stream_parameter_groups() method
        optimizer_type: Type of optimizer ('adamw', 'sgd', 'adam', etc.)
        stream_lrs: Dict mapping stream names to learning rates
                   Keys: 'stream1', 'stream2', 'stream3', ..., 'shared'
        stream_weight_decays: Dict mapping stream names to weight decay values
                             If None, defaults to 0.0 for all groups
        momentum: Momentum for SGD
        betas: Beta parameters for Adam/AdamW
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured PyTorch optimizer
    """
    # Get parameter groups from model
    param_groups = model.get_stream_parameter_groups(
        stream_lrs=stream_lrs,
        stream_weight_decays=stream_weight_decays
    )

    # Create optimizer with parameter groups
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, betas=betas, **kwargs)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=betas, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer
```

---

### 7.3 schedulers.py - Learning Rate Schedulers

**File**: `src/training/schedulers.py`

**Status**: ✅ **NO CHANGES NEEDED**

**Analysis**:
- All scheduler utilities are already generic and support arbitrary numbers of parameter groups
- `PerGroupSchedulerWrapper` wraps any scheduler and applies it per group
- `setup_scheduler()` supports list-based `eta_min` parameter for CosineAnnealingLR
- Custom schedulers (`DecayingScheduler`, `WarmupScheduler`) work with any number of groups

**Example usage (already works)**:
```python
# Works with N parameter groups out of the box
scheduler = setup_scheduler(
    optimizer,
    scheduler_type='cosine',
    eta_min=[1e-6, 1e-7, 1e-6],  # One per parameter group
    T_max=epochs
)
```

---

### 7.4 model_helpers.py - Training Utilities

**File**: `src/models/common/model_helpers.py`

**Current Issues**:
1. **Lines 206-272**: `setup_stream_early_stopping()` hardcoded with `stream1_patience` and `stream2_patience`
2. **Lines 274-480**: `check_stream_early_stopping()` hardcoded for `.stream1_` and `.stream2_` parameter patterns
3. Returns dicts with hardcoded `'stream1'` and `'stream2'` keys

**Required Changes**:

#### 7.4.1 setup_stream_early_stopping() - N-Stream Only

**CURRENT (2-stream only):**
```python
def setup_stream_early_stopping(
    stream_early_stopping: bool,
    stream_monitor: str = 'val_loss',
    stream1_patience: int = 10,
    stream2_patience: int = 10,
    stream_min_delta: float = 0.001,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Setup early stopping configuration for individual streams.

    Only supports 2 streams.
    """
    if not stream_early_stopping:
        return None

    config = {
        'enabled': True,
        'monitor': stream_monitor,
        'stream1': {
            'patience': stream1_patience,
            'min_delta': stream_min_delta,
            'counter': 0,
            'best_metric': float('inf') if 'loss' in stream_monitor else 0.0
        },
        'stream2': {
            'patience': stream2_patience,
            'min_delta': stream_min_delta,
            'counter': 0,
            'best_metric': float('inf') if 'loss' in stream_monitor else 0.0
        }
    }

    return config
```

**TARGET (N-stream, clean break):**
```python
def setup_stream_early_stopping(
    stream_early_stopping: bool,
    stream_patience: Dict[str, int],
    stream_monitor: str = 'val_loss',
    stream_min_delta: float = 0.001,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Setup early stopping configuration for individual streams.

    Usage:
        setup_stream_early_stopping(
            stream_early_stopping=True,
            stream_patience={'stream1': 10, 'stream2': 15, 'stream3': 20}
        )

    Args:
        stream_early_stopping: Whether to enable stream-specific early stopping
        stream_patience: Dict mapping stream names to patience values
                        Keys: 'stream1', 'stream2', 'stream3', ...
        stream_monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
        stream_min_delta: Minimum change to qualify as improvement
        verbose: Whether to print messages

    Returns:
        Configuration dict for stream early stopping, or None if disabled
    """
    if not stream_early_stopping:
        return None

    # Build configuration
    config = {
        'enabled': True,
        'monitor': stream_monitor,
        'min_delta': stream_min_delta,
        'verbose': verbose
    }

    # Add per-stream configs
    for stream_name, patience in stream_patience.items():
        config[stream_name] = {
            'patience': patience,
            'counter': 0,
            'best_metric': float('inf') if 'loss' in stream_monitor else 0.0,
            'frozen': False
        }

    return config
```

#### 7.4.2 check_stream_early_stopping() - N-Stream Dynamic Detection

**CURRENT (2-stream only):**
```python
def check_stream_early_stopping(
    model,
    stream_metrics: Dict[str, float],  # e.g., {'stream1_val_loss': 0.5, 'stream2_val_loss': 0.3}
    early_stopping_config: Dict[str, Any],
    epoch: int
) -> Dict[str, bool]:
    """
    Check if any streams should be frozen due to early stopping.

    Returns:
        Dict with 'stream1' and 'stream2' keys indicating if frozen
    """
    if not early_stopping_config or not early_stopping_config['enabled']:
        return {'stream1': False, 'stream2': False}

    monitor = early_stopping_config['monitor']
    min_delta = early_stopping_config['min_delta']

    frozen_status = {}

    # Check stream 1
    stream1_metric = stream_metrics.get(f'stream1_{monitor}')
    if stream1_metric is not None:
        stream1_config = early_stopping_config['stream1']
        # ... early stopping logic ...
        frozen_status['stream1'] = should_freeze_stream1

    # Check stream 2
    stream2_metric = stream_metrics.get(f'stream2_{monitor}')
    if stream2_metric is not None:
        stream2_config = early_stopping_config['stream2']
        # ... early stopping logic ...
        frozen_status['stream2'] = should_freeze_stream2

    return frozen_status
```

**TARGET (N-stream, clean break):**
```python
def check_stream_early_stopping(
    model,
    stream_metrics: Dict[str, float],
    early_stopping_config: Dict[str, Any],
    epoch: int
) -> Dict[str, bool]:
    """
    Check if any streams should be frozen due to early stopping.

    Dynamically detects configured streams from early_stopping_config.

    Args:
        model: The model (may need to freeze stream parameters)
        stream_metrics: Dict of metrics for each stream
                       e.g., {'stream1_val_loss': 0.5, 'stream2_val_loss': 0.3, 'stream3_val_loss': 0.4}
        early_stopping_config: Config from setup_stream_early_stopping()
        epoch: Current epoch number

    Returns:
        Dict mapping stream names to frozen status (True if should freeze)
        e.g., {'stream1': False, 'stream2': True, 'stream3': False}
    """
    if not early_stopping_config or not early_stopping_config['enabled']:
        return {}

    monitor = early_stopping_config['monitor']
    min_delta = early_stopping_config['min_delta']
    verbose = early_stopping_config.get('verbose', True)

    frozen_status = {}

    # Iterate over all configured streams
    for stream_name in early_stopping_config.keys():
        # Skip special config keys
        if stream_name in ['enabled', 'monitor', 'min_delta', 'verbose']:
            continue

        stream_config = early_stopping_config[stream_name]

        # Skip if already frozen
        if stream_config.get('frozen', False):
            frozen_status[stream_name] = True
            continue

        # Get metric for this stream
        metric_key = f'{stream_name}_{monitor}'
        current_metric = stream_metrics.get(metric_key)

        if current_metric is None:
            # Metric not available, don't freeze
            frozen_status[stream_name] = False
            continue

        # Check if improved
        best_metric = stream_config['best_metric']
        is_loss_metric = 'loss' in monitor

        if is_loss_metric:
            improved = (best_metric - current_metric) > min_delta
        else:
            improved = (current_metric - best_metric) > min_delta

        if improved:
            # Update best and reset counter
            stream_config['best_metric'] = current_metric
            stream_config['counter'] = 0
            frozen_status[stream_name] = False
        else:
            # No improvement, increment counter
            stream_config['counter'] += 1

            # Check if patience exhausted
            if stream_config['counter'] >= stream_config['patience']:
                if verbose:
                    print(f"[Epoch {epoch}] Early stopping triggered for {stream_name} "
                          f"(patience {stream_config['patience']} exhausted)")
                stream_config['frozen'] = True
                frozen_status[stream_name] = True
            else:
                frozen_status[stream_name] = False

    return frozen_status
```

---

## Summary of Support Files Updates

| File | Status | Changes Required | Notes |
|------|--------|------------------|-------|
| **abstract_model.py** | ⚠️ Needs update | Update `get_stream_parameter_groups()` to dict-based API, add generic `_forward_stream_pathway()`, update `forward()` signature to accept `List[Tensor]` | Clean break - no backward compatibility |
| **optimizers.py** | ⚠️ Needs update | Update `create_stream_optimizer()` to dict-based API | Clean break - no backward compatibility |
| **schedulers.py** | ✅ No changes | Already generic, supports N parameter groups | No changes needed |
| **model_helpers.py** | ⚠️ Needs update | Update `setup_stream_early_stopping()` and `check_stream_early_stopping()` to dict-based API with dynamic stream detection | Clean break - no backward compatibility |

**Refactoring Strategy**: **Clean Break (Option A)** - All support files will be refactored to only support the new N-stream API. Legacy 2-stream code will be updated separately to use the new dict-based API. This ensures cleaner, more maintainable code without the complexity of dual APIs.

---

## Summary

This refactoring converts LINet3 from a hardcoded 2-stream architecture to a flexible N-stream design while preserving the core integration mechanism. Key changes:

1. **Replace hardcoded parameters** with `nn.ParameterList` for dynamic stream handling
2. **Update method signatures** to accept `List[Tensor]` instead of individual stream arguments
3. **Preserve integration logic**: Linear weighted sum of all stream outputs
4. **Clean break approach**: New N-stream API with dict-based configuration (no backward compatibility with old 2-stream signatures)
5. **Update monitoring/utilities** for dynamic stream detection
6. **Update support files** with new dict-based APIs for N-stream configuration
7. **Update factory functions** (`li_resnet18`, etc.) to accept `stream_input_channels` as a list
8. **Refactor auxiliary classifiers** from `fc_stream1`, `fc_stream2` to `fc_streams` ModuleList
9. **Consolidate stream pathway methods** from separate `_forward_stream1_pathway()` and `_forward_stream2_pathway()` into single `_forward_stream_pathway(stream_idx, stream_input)`

The refactoring is systematic, touching:
- **7 files in li_net3 module**: conv.py, pooling.py, container.py, blocks.py, li_net.py, gradient_monitor.py, stream_monitor.py
- **3 support files**: abstract_model.py, optimizers.py, model_helpers.py
- **All factory functions**: li_resnet18, li_resnet34, li_resnet50, li_resnet101, li_resnet152

This enables future extensibility for additional data streams beyond RGB, Depth, and Orthogonal. Legacy 2-stream code will be updated separately to use the new API.
