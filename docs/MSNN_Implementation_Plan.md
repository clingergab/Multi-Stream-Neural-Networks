# Multi-Stream Neural Networks - Implementation Plan

**Project Goal**: Implement and validate Multi-Stream Neural Networks (MSNNs) for RGB-D scene classification on NYU Depth V2 dataset.

---

## 1. Project Structure

```
msnn-project/
├── data/
│   ├── raw/                    # NYU Depth V2 dataset
│   ├── processed/              # Preprocessed train/test splits
│   └── dataloader.py           # Data loading utilities
├── models/
│   ├── baseline.py             # RGB-only, Depth-only, Early/Late Fusion
│   ├── approach1_basic.py      # Basic Multi-Channel
│   ├── approach2_concat.py     # Concat + Linear Integration
│   ├── approach3_direct.py     # Direct Mixing (α, β, γ)
│   └── resnet_backbone.py      # ResNet18 building blocks
├── training/
│   ├── train.py                # Main training loop
│   ├── eval.py                 # Evaluation scripts
│   └── utils.py                # Training utilities
├── experiments/
│   ├── configs/                # YAML config files for each experiment
│   └── results/                # Training logs, checkpoints, metrics
├── analysis/
│   ├── visualize_weights.py    # Integration weight analysis
│   └── robustness_tests.py     # Modality corruption tests
├── requirements.txt
└── README.md
```

---

## 2. Dataset: NYU Depth V2

### 2.1 Dataset Specifications

- **Source**: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- **Task**: 40-class indoor scene classification
- **Size**: 1,449 RGB-D images
  - Training: ~795 images
  - Testing: ~654 images
- **Image Format**: 
  - RGB: 640×480 pixels, 3 channels
  - Depth: 640×480 pixels, 1 channel (depth in meters)

### 2.2 Data Preprocessing

```python
# Preprocessing pipeline
1. Resize images to 224×224 (ImageNet standard for ResNet)
2. Normalize RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
3. Normalize Depth: 
   - Clip to valid range (0.5m to 10m)
   - Convert to log scale: log(depth + 1)
   - Normalize to [0, 1]
4. Data Augmentation (training only):
   - Random horizontal flip (p=0.5)
   - Random crop (224×224 from 256×256)
   - Color jitter: brightness=0.2, contrast=0.2, saturation=0.2
   - Depth noise: Gaussian noise (σ=0.05)
```

### 2.3 Dataloader Implementation

```python
class NYUDepthDataset(torch.utils.data.Dataset):
    """
    Returns:
        rgb: Tensor [3, 224, 224]
        depth: Tensor [1, 224, 224]
        label: int (0-39)
    """
    def __init__(self, split='train', transform=None):
        # Load image paths and labels
        # Apply transforms
        pass
    
    def __getitem__(self, idx):
        # Load RGB and Depth
        # Apply preprocessing
        # Return (rgb, depth, label)
        pass

# DataLoader settings
batch_size = 32
num_workers = 4
shuffle = True (for train), False (for test)
```

---

## 3. Model Architectures

### 3.1 Baseline Models

#### 3.1.1 RGB-Only ResNet18
```python
class RGBOnlyModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # Standard ResNet18
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, rgb, depth):
        # Ignore depth input
        return self.resnet(rgb)
```

#### 3.1.2 Depth-Only ResNet18
```python
class DepthOnlyModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # Modify first conv for 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, rgb, depth):
        # Ignore rgb input
        return self.resnet(depth)
```

#### 3.1.3 Early Fusion ResNet18
```python
class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        # Modify first conv for 4-channel input (RGB + Depth)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, rgb, depth):
        # Concatenate RGB and Depth
        x = torch.cat([rgb, depth], dim=1)  # [B, 4, 224, 224]
        return self.resnet(x)
```

#### 3.1.4 Late Fusion ResNet18
```python
class LateFusionModel(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # Two separate ResNet18 models
        self.rgb_model = RGBOnlyModel(num_classes)
        self.depth_model = DepthOnlyModel(num_classes)
    
    def forward(self, rgb, depth):
        rgb_logits = self.rgb_model(rgb, None)
        depth_logits = self.depth_model(None, depth)
        # Average predictions
        return (rgb_logits + depth_logits) / 2
```

**Expected Baseline Performance**:
- RGB-Only: 55-58%
- Depth-Only: 45-50%
- Early Fusion: 60-63%
- Late Fusion: 62-65%

---

### 3.2 Multi-Stream Approach 1: Basic Multi-Channel

**Concept**: Separate RGB and Depth pathways remain independent until final concatenation.

#### Architecture Specification

```python
class BasicMultiChannelModel(nn.Module):
    """
    Two separate ResNet18 backbones with shared architecture but independent weights.
    Features concatenated before final classification layer.
    """
    def __init__(self, num_classes=40):
        super().__init__()
        
        # RGB pathway - uses ImageNet pretrained weights
        self.rgb_backbone = self._build_resnet18_backbone(in_channels=3, pretrained=True)
        
        # Depth pathway - random initialization
        self.depth_backbone = self._build_resnet18_backbone(in_channels=1, pretrained=False)
        
        # Classifier on concatenated features
        # ResNet18 outputs 512 features per pathway
        self.classifier = nn.Linear(512 + 512, num_classes)
    
    def _build_resnet18_backbone(self, in_channels, pretrained):
        """
        Returns ResNet18 backbone without final FC layer
        """
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        # Modify first conv if needed
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove final FC layer
        resnet.fc = nn.Identity()
        
        return resnet
    
    def forward(self, rgb, depth):
        # Process each pathway independently
        rgb_features = self.rgb_backbone(rgb)      # [B, 512]
        depth_features = self.depth_backbone(depth) # [B, 512]
        
        # Concatenate features
        combined_features = torch.cat([rgb_features, depth_features], dim=1)  # [B, 1024]
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits
```

**Key Properties**:
- Parameters: ~23.4M (11.7M per pathway + 0.04M classifier)
- No cross-modal learning during feature extraction
- Baseline for measuring integration benefits

---

### 3.3 Multi-Stream Approach 2: Concat + Linear Integration

**Concept**: Features from both pathways are concatenated and transformed at each layer to create an integrated stream.

#### Architecture Specification

```python
class ConcatLinearLayer(nn.Module):
    """
    Single layer with RGB, Depth, and Integrated streams.
    Integration through learned linear transformation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Separate pathway convolutions
        self.rgb_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.rgb_bn = nn.BatchNorm2d(out_channels)
        
        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.depth_bn = nn.BatchNorm2d(out_channels)
        
        # Integration layer
        # Input: concatenated RGB + Depth + previous integrated (if exists)
        # For first layer: only RGB + Depth (2 * out_channels)
        # For subsequent layers: RGB + Depth + Integrated (3 * out_channels)
        self.integration_conv_first = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        self.integration_conv = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)
        self.integration_bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_in, depth_in, integrated_in=None):
        # Process RGB pathway
        rgb_out = self.rgb_conv(rgb_in)
        rgb_out = self.rgb_bn(rgb_out)
        rgb_out = self.relu(rgb_out)
        
        # Process Depth pathway
        depth_out = self.depth_conv(depth_in)
        depth_out = self.depth_bn(depth_out)
        depth_out = self.relu(depth_out)
        
        # Integration
        if integrated_in is None:
            # First layer: only RGB + Depth
            concat_features = torch.cat([rgb_out, depth_out], dim=1)
            integrated_out = self.integration_conv_first(concat_features)
        else:
            # Subsequent layers: RGB + Depth + previous integrated
            concat_features = torch.cat([rgb_out, depth_out, integrated_in], dim=1)
            integrated_out = self.integration_conv(concat_features)
        
        integrated_out = self.integration_bn(integrated_out)
        integrated_out = self.relu(integrated_out)
        
        return rgb_out, depth_out, integrated_out


class ConcatLinearModel(nn.Module):
    """
    Full network with Concat + Linear Integration
    """
    def __init__(self, num_classes=40):
        super().__init__()
        
        # Initial convolutions (separate for RGB and Depth)
        self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_bn1 = nn.BatchNorm2d(64)
        
        self.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style layers with integration
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier uses only integrated features
        self.classifier = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First block handles stride and channel change
        layers.append(ConcatLinearLayer(in_channels, out_channels))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ConcatLinearLayer(out_channels, out_channels))
        return nn.ModuleList(layers)
    
    def forward(self, rgb, depth):
        # Initial convolutions
        rgb = self.rgb_conv1(rgb)
        rgb = self.rgb_bn1(rgb)
        rgb = self.relu(rgb)
        
        depth = self.depth_conv1(depth)
        depth = self.depth_bn1(depth)
        depth = self.relu(depth)
        
        # Maxpool
        rgb = self.maxpool(rgb)
        depth = self.maxpool(depth)
        
        integrated = None
        
        # Layer 1
        for layer in self.layer1:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        # Layer 2
        for layer in self.layer2:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        # Layer 3
        for layer in self.layer3:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        # Layer 4
        for layer in self.layer4:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        # Use only integrated features for classification
        x = self.avgpool(integrated)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
```

**Key Properties**:
- Parameters: ~25-30M (additional integration layers)
- Cross-modal learning at every layer
- Standard gradient flow through concatenated features

---

### 3.4 Multi-Stream Approach 3: Direct Mixing (α, β, γ)

**Concept**: Learnable scalar parameters control pathway mixing at each layer.

#### Architecture Specification

```python
class DirectMixingLayer(nn.Module):
    """
    Layer with learnable integration weights (α, β, γ).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Separate pathway convolutions
        self.rgb_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.rgb_bn = nn.BatchNorm2d(out_channels)
        
        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.depth_bn = nn.BatchNorm2d(out_channels)
        
        # Learnable integration weights (scalar per layer)
        self.alpha = nn.Parameter(torch.tensor(1.0))  # RGB weight
        self.beta = nn.Parameter(torch.tensor(1.0))   # Depth weight
        self.gamma = nn.Parameter(torch.tensor(0.2))  # Previous integrated weight
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, rgb_in, depth_in, integrated_in=None):
        # Process RGB pathway
        rgb_out = self.rgb_conv(rgb_in)
        rgb_out = self.rgb_bn(rgb_out)
        rgb_out = self.relu(rgb_out)
        
        # Process Depth pathway
        depth_out = self.depth_conv(depth_in)
        depth_out = self.depth_bn(depth_out)
        depth_out = self.relu(depth_out)
        
        # Direct mixing with learnable weights
        # Apply clamping to prevent pathway collapse
        alpha_reg = torch.clamp(self.alpha, min=0.01)
        beta_reg = torch.clamp(self.beta, min=0.01)
        gamma_reg = torch.clamp(self.gamma, min=0.01)
        
        if integrated_in is None:
            # First layer: only RGB + Depth
            integrated_out = alpha_reg * rgb_out + beta_reg * depth_out
        else:
            # Subsequent layers: RGB + Depth + previous integrated
            integrated_out = (alpha_reg * rgb_out + 
                            beta_reg * depth_out + 
                            gamma_reg * integrated_in)
        
        return rgb_out, depth_out, integrated_out


class DirectMixingModel(nn.Module):
    """
    Full network with Direct Mixing integration (α, β, γ)
    """
    def __init__(self, num_classes=40):
        super().__init__()
        
        # Initial convolutions
        self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_bn1 = nn.BatchNorm2d(64)
        
        self.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_bn1 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style layers with direct mixing
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(DirectMixingLayer(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(DirectMixingLayer(out_channels, out_channels))
        return nn.ModuleList(layers)
    
    def forward(self, rgb, depth):
        # Initial convolutions
        rgb = self.rgb_conv1(rgb)
        rgb = self.rgb_bn1(rgb)
        rgb = self.relu(rgb)
        
        depth = self.depth_conv1(depth)
        depth = self.depth_bn1(depth)
        depth = self.relu(depth)
        
        # Maxpool
        rgb = self.maxpool(rgb)
        depth = self.maxpool(depth)
        
        integrated = None
        
        # Process through layers
        for layer in self.layer1:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        for layer in self.layer2:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        for layer in self.layer3:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        for layer in self.layer4:
            rgb, depth, integrated = layer(rgb, depth, integrated)
        
        # Classification from integrated features
        x = self.avgpool(integrated)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_integration_weights(self):
        """
        Extract α, β, γ weights for analysis.
        Returns dict mapping layer names to (α, β, γ) tuples.
        """
        weights = {}
        for name, module in self.named_modules():
            if isinstance(module, DirectMixingLayer):
                weights[name] = (
                    module.alpha.item(),
                    module.beta.item(),
                    module.gamma.item()
                )
        return weights
```

**Key Properties**:
- Parameters: ~23.4M + 3 scalars per layer (minimal overhead)
- High interpretability - can visualize α, β, γ evolution
- Self-organizing mixing ratios

---

## 4. Training Configuration

### 4.1 Hyperparameters

```yaml
# Base configuration (applies to all models)
training:
  epochs: 100
  batch_size: 32
  num_workers: 4
  
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
  
  scheduler:
    type: "ReduceLROnPlateau"
    mode: "max"  # maximize validation accuracy
    factor: 0.5
    patience: 10
    min_lr: 0.00001
  
  loss:
    type: "CrossEntropyLoss"
  
  early_stopping:
    patience: 20
    metric: "val_accuracy"

# Gradient clipping for multi-stream models
gradient_clipping:
  enabled: true
  max_norm: 1.0

# Integration weight regularization (for Approach 3)
integration_regularization:
  enabled: true
  min_weight: 0.01  # Minimum value for α, β, γ (prevent collapse)
```

### 4.2 Training Loop Structure

```python
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Single training epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (rgb, depth, labels) in enumerate(train_loader):
        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(rgb, depth)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (if enabled)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validation/Testing.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for rgb, depth, labels in val_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
            
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main_training_loop(model, train_loader, val_loader, config, device):
    """
    Complete training loop with early stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Logging
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc
```

---

## 5. Evaluation Protocol

### 5.1 Primary Metrics

```python
def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for rgb, depth, labels in test_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            outputs = model(rgb, depth)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
```

### 5.2 Robustness Testing

```python
def test_modality_corruption(model, test_loader, device, corruption_type='rgb_noise'):
    """
    Test robustness to modality-specific corruption.
    
    Corruption types:
    - 'rgb_noise': Add Gaussian noise to RGB
    - 'rgb_brightness': Random brightness shifts
    - 'depth_noise': Add noise to depth
    - 'depth_missing': Randomly set depth regions to 0
    - 'rgb_missing': Test with RGB = 0 (depth only)
    - 'depth_missing_full': Test with Depth = 0 (RGB only)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for rgb, depth, labels in test_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
            
            # Apply corruption
            if corruption_type == 'rgb_noise':
                rgb = rgb + torch.randn_like(rgb) * 0.1
            elif corruption_type == 'rgb_brightness':
                rgb = rgb * (1 + torch.randn(rgb.size(0), 1, 1, 1).to(device) * 0.3)
            elif corruption_type == 'depth_noise':
                depth = depth + torch.randn_like(depth) * 0.05
            elif corruption_type == 'depth_missing':
                mask = torch.rand_like(depth) > 0.3  # 30% missing
                depth = depth * mask
            elif corruption_type == 'rgb_missing_full':
                rgb = torch.zeros_like(rgb)
            elif corruption_type == 'depth_missing_full':
                depth = torch.zeros_like(depth)
            
            outputs = model(rgb, depth)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy
```

### 5.3 Integration Weight Analysis (Approach 3)

```python
def analyze_integration_weights(model, save_path='integration_weights.png'):
    """
    Visualize evolution of α, β, γ across layers.
    Only applicable to DirectMixingModel.
    """
    if not isinstance(model, DirectMixingModel):
        print("Integration weight analysis only for DirectMixingModel")
        return
    
    weights = model.get_integration_weights()
    
    # Extract values
    layer_names = list(weights.keys())
    alphas = [weights[name][0] for name in layer_names]
    betas = [weights[name][1] for name in layer_names]
    gammas = [weights[name][2] for name in layer_names]
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(alphas, 'r-o', label='α (RGB)')
    axes[0].set_title('RGB Weight (α) Across Layers')
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Weight Value')
    axes[0].grid(True)
    
    axes[1].plot(betas, 'b-o', label='β (Depth)')
    axes[1].set_title('Depth Weight (β) Across Layers')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Weight Value')
    axes[1].grid(True)
    
    axes[2].plot(gammas, 'g-o', label='γ (Integrated)')
    axes[2].set_title('Integrated Weight (γ) Across Layers')
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Weight Value')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Integration weight visualization saved to {save_path}")
```

---

## 6. Experimental Workflow

### 6.1 Experiment Sequence

```
1. Baseline Experiments (Week 1)
   - RGB-Only
   - Depth-Only
   - Early Fusion
   - Late Fusion
   Goal: Establish performance range (55-65%)

2. MSNN Approach 1 (Week 2)
   - Basic Multi-Channel
   Goal: Match or exceed Late Fusion (62-65%)
   Measure: Training speedup vs Late Fusion

3. MSNN Approach 2 (Week 3)
   - Concat + Linear Integration
   Goal: Improve over Approach 1 through cross-modal learning
   Measure: Accuracy improvement, training time

4. MSNN Approach 3 (Week 4)
   - Direct Mixing (α, β, γ)
   Goal: 66-69% accuracy with high interpretability
   Measure: Accuracy, integration weight patterns

5. Robustness Analysis (Week 5)
   - Test all models under corruption
   - Compare robustness across approaches

6. Analysis & Documentation (Week 6)
   - Statistical significance tests
   - Visualization of results
   - Write up findings
```

### 6.2 Success Criteria

**Minimum Success**:
- Approach 1 matches Late Fusion (62-65%) with 1.3x speedup
- Clear gradient flow in both pathways (no collapse)

**Target Success**:
- Any MSNN approach achieves 66-69% accuracy
- 1.5-2x training speedup over Late Fusion
- Integration weights show interpretable patterns (Approach 3)

**Stretch Goals**:
- Exceed 69% accuracy
- Demonstrate superior robustness to modality corruption
- Clear modality importance patterns at different network depths

---

## 7. Results Logging and Tracking

### 7.1 Experiment Tracking Structure

```python
# Use tensorboard or wandb for logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/{experiment_name}')

# Log per epoch
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

# For Approach 3: Log integration weights
if isinstance(model, DirectMixingModel):
    weights = model.get_integration_weights()
    for name, (alpha, beta, gamma) in weights.items():
        writer.add_scalar(f'Integration/{name}/alpha', alpha, epoch)
        writer.add_scalar(f'Integration/{name}/beta', beta, epoch)
        writer.add_scalar(f'Integration/{name}/gamma', gamma, epoch)
```

### 7.2 Results Summary Format

```python
# Save final results
results = {
    'model_name': 'DirectMixingModel',
    'test_accuracy': 67.3,
    'train_time_minutes': 180,
    'total_parameters': 23_400_000,
    'best_epoch': 45,
    'final_lr': 0.0001,
    'robustness': {
        'rgb_noise': 64.2,
        'depth_noise': 65.8,
        'rgb_missing': 52.1,
        'depth_missing': 58.9
    },
    'integration_weights': {
        'layer1': {'alpha': 1.2, 'beta': 0.8, 'gamma': 0.3},
        # ... all layers
    }
}

import json
with open('results/approach3_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 8. Implementation Priority Order

### Phase 1: Foundation (Priority 1)
1. Dataset loading and preprocessing
2. RGB-Only baseline
3. Basic training loop
4. Evaluation metrics

### Phase 2: Baselines (Priority 2)
5. Depth-Only baseline
6. Early Fusion baseline
7. Late Fusion baseline

### Phase 3: Core MSNN (Priority 3)
8. Approach 1: Basic Multi-Channel
9. Approach 3: Direct Mixing (most interesting for interpretability)
10. Approach 2: Concat + Linear (if time permits)

### Phase 4: Analysis (Priority 4)
11. Robustness testing
12. Integration weight visualization
13. Statistical analysis

---

## 9. Key Technical Notes

### 9.1 Common Pitfalls to Avoid

1. **Pathway Collapse**: Monitor gradient norms per pathway. If one pathway's gradients are consistently 10x smaller, investigate.

2. **Memory Issues**: NYU Depth V2 depth maps can be large. Ensure proper resizing before batching.

3. **Normalization**: RGB and Depth have different scales. Always normalize separately.

4. **Overfitting**: Dataset is small (1,449 images). Use strong augmentation and early stopping.

5. **Integration Weight Initialization**: For Approach 3, don't initialize α, β, γ all to 0. Use 1.0, 1.0, 0.2.

### 9.2 Debugging Checklist

```python
# Before full training, verify:
1. Data loader returns correct shapes: rgb [B,3,224,224], depth [B,1,224,224], labels [B]
2. Model forward pass works: output shape [B, 40]
3. Loss computation works: scalar value
4. Backward pass works: gradients exist for all parameters
5. Single batch overfitting: Can model overfit to 1 batch? (should reach ~100% accuracy)

# Debug script
def debug_model(model, dataloader, device):
    rgb, depth, labels = next(iter(dataloader))
    rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
    
    print(f"Input shapes: RGB {rgb.shape}, Depth {depth.shape}, Labels {labels.shape}")
    
    outputs = model(rgb, depth)
    print(f"Output shape: {outputs.shape}")
    
    loss = nn.CrossEntropyLoss()(outputs, labels)
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    print("Backward pass successful")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
```

---

## 10. Expected Outputs

### 10.1 Final Deliverables

1. **Trained Models**: Checkpoints for all approaches
2. **Results Summary**: JSON files with all metrics
3. **Visualizations**:
   - Training curves (loss, accuracy)
   - Confusion matrices
   - Integration weight evolution (Approach 3)
   - Robustness comparison plots
4. **Code Repository**: Clean, documented, reproducible
5. **Analysis Report**: Summary of findings, comparison table

### 10.2 Comparison Table (Target)

| Model | Params | Train Time | Test Acc | RGB Corrupt | Depth Corrupt |
|-------|--------|------------|----------|-------------|---------------|
| RGB-Only | 11.7M | 1x | 56% | 52% | 56% |
| Depth-Only | 11.7M | 1x | 48% | 48% | 42% |
| Early Fusion | 11.7M | 1x | 62% | 58% | 59% |
| Late Fusion | 23.4M | 2x | 64% | 60% | 61% |
| **MSNN Approach 1** | 23.4M | 1.3x | 65% | 61% | 62% |
| **MSNN Approach 2** | 26M | 1.4x | 67% | 63% | 64% |
| **MSNN Approach 3** | 23.4M | 1.3x | 68% | 64% | 65% |

---

## 11. Next Steps After Implementation

1. **Scale to Larger Architectures**: Try ResNet50, test if principles hold
2. **Transfer Learning**: Test on SUN RGB-D dataset
3. **Additional Modalities**: Extend to RGB + Depth + Thermal (3 streams)
4. **Architecture Variants**: 
   - Channel-wise adaptive integration (per-channel α, β, γ)
   - Dynamic input-dependent integration
5. **Deployment**: Optimize for inference, measure real-world speedup

---

**End of Implementation Plan**

This document provides all technical specifications needed to implement and validate Multi-Stream Neural Networks. Follow the priority order, use the debugging checklist, and track all experiments systematically.
