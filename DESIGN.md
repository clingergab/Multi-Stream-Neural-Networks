# **Multi-Stream Neural Networks**

## **Executive Summary**

This research proposal introduces Multi-Stream Neural Networks (MSNNs), a novel neural architecture inspired by biological visual processing that employs multiple weight connections between neurons to separately process distinct features like color and brightness. Unlike traditional neural networks that use a single weight to aggregate all information between neurons, MSNNs use specialized weight channels that can develop expertise in particular domains of visual information, potentially yielding improved performance on image recognition tasks.

## **Background & Motivation**

### **Biological Inspiration**

The human visual system processes different aspects of visual information through specialized pathways:

* The parvocellular pathway primarily processes color and form  
* The magnocellular pathway primarily processes brightness and motion  
* These pathways maintain some separation while also integrating at higher levels

Traditional ANNs use single weights between neurons, forcing all visual information (color, brightness, texture, etc.) through the same connection. This differs from biological neural networks where multiple synaptic connections between the same neurons allow for parallel processing of different information types.

### **Current Limitations**

While convolutional neural networks (CNNs) have achieved impressive results in image recognition, they still face challenges in:

* Distinguishing objects with similar shapes but different colors  
* Recognizing objects under varying lighting conditions  
* Generalizing across color and brightness variations

## **Proposed Architecture**

### **Core Concept**

The Multi-Stream Neural Network architecture introduces multiple parallel weights between each pair of connected neurons, with each weight specializing in different aspects of visual information:

1. **Color Weights (wc)**: Specialized for processing color information  
2. **Brightness Weights (wb)**: Specialized for processing brightness/luminance information  
3. **Optional additional weights**: Could include texture, edge, or motion-specific weights

## **![][image1]**

## **Implementation Approaches**

**Solving the Information Mixing Problem**

A key challenge: After the first layer, how do we prevent color and brightness information from getting all mixed together? Our approaches can be grouped by their fundamental architectural strategy:

### **1 Basic Multi-Channel Neurons**

Each neuron outputs separate values for each modality:

**Mathematical Formulation:**

y\_color \= f(Σ(wc\_i \* xc\_i) \+ b\_c)  
y\_brightness \= f(Σ(wb\_i \* xb\_i) \+ b\_b)  
output \= \[y\_color, y\_brightness\]

**Implementation:**

| class MultiChannelLayer:    def \_\_init\_\_(self, input\_size):        self.color\_weights \= initialize\_weights(input\_size)        self.brightness\_weights \= initialize\_weights(input\_size)        self.bias \= initialize\_bias()            def forward(self, color\_inputs, brightness\_inputs):        color\_output \= activation\_fn(dot(self.color\_weights, color\_inputs) \+ self.bias)        brightness\_output \= activation\_fn(dot(self.brightness\_weights, brightness\_inputs) \+ self.bias)        return \[color\_output, brightness\_output\] class MultiWeightNetwork(nn.Module):    def \_\_init\_\_(self, input\_size, hidden\_size, num\_classes):        super().\_\_init\_\_()        self.layer1 \= MultiWeightLayer(input\_size\*3, input\_size, hidden\_size)        self.layer2 \= MultiWeightLayer(hidden\_size, hidden\_size, hidden\_size)        self.layer3 \= MultiWeightLayer(hidden\_size, hidden\_size, hidden\_size)                \# Only integrated features go to classifier        self.classifier \= nn.Linear(hidden\_size, num\_classes)            def forward(self, color, brightness):                \# Each layer: two inputs → two outputs        c1, b1 \= self.layer1(color, brightness)        c2, b2 \= self.layer2(c1, b1)  \# Uses separate streams        c3, b3 \= self.layer3(c2, b2)  \# Still separate                combined \= torch.cat(\[c3, b3\], dim\=1) *\# Simple concatenation*         return self.classifier(combined) |
| :---- |

##### 

#### Loss and Backpropagation Design

In the basic multi-channel model, color and brightness pathways remain separate throughout the network, combining only at the final classification layer. This design maintains complete pathway independence while allowing both modalities to contribute to the final prediction.

##### 

**Loss Calculation**

*\# Single loss on final combined output*

predictions \= classifier(concat(\[color\_features, brightness\_features\]))

loss \= CrossEntropyLoss(predictions, labels)

**Backpropagation Calculation**

The gradient flows back through the concatenation point and automatically splits to update both pathways:

∂L/∂color\_weights \= ∂L/∂predictions × ∂predictions/∂color\_features × ∂color\_features/∂color\_weights

∂L/∂brightness\_weights \= ∂L/∂predictions × ∂predictions/∂brightness\_features × ∂brightness\_features/∂brightness\_weights

#### Gradient Flow Visualization

                   Loss

                     ↓

                Classifier

               /            \\

              /               \\

\[Color Features\]   \[Brightness Features\]

         ↓                    ↓

    Color Pathway       Brightness Pathway

         ↓                    ↓

    Color Weights       Brightness Weights

Each pathway receives gradients independently based on its contribution to the final prediction. The pathways learn in parallel without direct interaction.

### 2 Continuous Learnable Integration 

The continuous integration model extends the basic multi-channel architecture by introducing a third information stream \- the integrated features \- that accumulates information as it flows through the network. Each neuron maintains separate feature processing while learning optimal integration strategies. The continuous integration model introduces learnable integration weights that determine how color and brightness features combine. While pathways remain separate, they jointly influence integrated features that flow to the next layer, creating a sophisticated interdependence.

[Continuous Integration Model: Implementation Approaches](https://docs.google.com/document/d/1pz3Uzac_yQ7DlcaMdDwrWoCEiw8HDgK-V6usLcwBZUY/edit?usp=sharing)

y\_color \= f(Σ(wc\_i \* xc\_i) \+ b\_c)  
y\_brightness \= f(Σ(wb\_i \* xb\_i) \+ b\_b)  
y\_integrated \= f(wi\_c \* y\_color \+ wi\_b \* y\_brightness)  
output \= \[y\_color, y\_brightness, y\_integrated\]

Where wi\_c and wi\_b are learnable integration weights.

**Key Advantages:**

* Color and brightness weights specialize on their respective features  
* Integration weights learn the optimal combination strategy for each task  
* All three sets of weights are continuously updated throughout training  
* The network learns task-specific integration rather than using fixed combination functions

### **2.1 Concatenation \+ Linear Integration**

#### Design Overview

This approach follows standard neural network patterns by concatenating all features and using learned linear transformations for integration. It provides maximum flexibility at the cost of interpretability. This is similar to how features are combined in ResNet, DenseNet, and other modern architectures.

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    ↓                    ↓  
\[W\_c¹\]                \[W\_b¹\]  
    ↓                    ↓  
Color₁               Brightness₁  
    ↓                    ↓  
             ↓  
        \[Concat: C₁,B₁\]  
             ↓  
          \[W\_i¹\]  
             ↓  
        Integrated₁  
             ↓  
    ↓                    ↓  
\[W\_c²\]                \[W\_b²\]  
    ↓                    ↓  
Color₂               Brightness₂  
    ↘               ↓          ↙  
             ↓  
        \[Concat: C₂,B₂,I₁\]  
             ↓  
          \[W\_i²\]  
             ↓  
        Integrated₂  
             ↓  
        (continues)  
             ↓  
        Integrated\_N  
             ↓  
        Classifier  
             ↓  
        Predictions

C\_l \= ReLU(W\_c^l · C\_{l-1} \+ b\_c^l)  
B\_l \= ReLU(W\_b^l · B\_{l-1} \+ b\_b^l)  
Z\_l \= \[C\_l; B\_l; I\_{l-1}\]  (concatenation)  
I\_l \= ReLU(W\_i^l · Z\_l \+ b\_i^l)

Where:

* `C_l, B_l`: Color and brightness features at layer l  
* `I_l`: Integrated features at layer l  
* `W_c^l, W_b^l, W_i^l`: Weight matrices  
* `Z_l`: Concatenated feature vector of dimension 3h

#### Implementation

| class ConcatLinearLayer(nn.Module):    def \_\_init\_\_(self, color\_size, brightness\_size, hidden\_size):        super().\_\_init\_\_()        \# Separate pathways        self.color\_weights \= nn.Linear(color\_size, hidden\_size)        self.brightness\_weights \= nn.Linear(brightness\_size, hidden\_size)                \# Integration through learned linear transformation        self.integration\_layer \= nn.Linear(hidden\_size \* 3, hidden\_size)  \# For 3 inputs        self.integration\_layer\_initial \= nn.Linear(hidden\_size \* 2, hidden\_size)  \# For 2 inputs            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways independently        color\_output \= F.relu(self.color\_weights(color\_input))        brightness\_output \= F.relu(self.brightness\_weights(brightness\_input))                \# Concatenate and transform        if prev\_integrated is not None:            combined \= torch.cat(\[color\_output, brightness\_output, prev\_integrated\], dim=1)            integrated\_output \= F.relu(self.integration\_layer(combined))        else:            combined \= torch.cat(\[color\_output, brightness\_output\], dim=1)            integrated\_output \= F.relu(self.integration\_layer\_initial(combined))                    return color\_output, brightness\_output, integrated\_output |
| :---- |

#### Backpropagation 

                   Loss  
                     ↓ ∂L/∂predictions  
                Classifier  
                     ↓ ∂L/∂I\_N  
              Integration\_N  
                     ↓ ∂L/∂I\_N  
                  \[W\_i^N\]  
                     ↓ ∂L/∂Z\_N \= W\_i^{N,T} · ∂L/∂I\_N  
            \[Concatenation Split\]  
         ↙           ↓            ↘  
    ∂L/∂C\_N     ∂L/∂B\_N      ∂L/∂I\_{N-1}  
        ↓           ↓              ↓  
    \[W\_c^N\]     \[W\_b^N\]    Integration\_{N-1}  
        ↓           ↓              ↓  
    ∂L/∂C\_{N-1} ∂L/∂B\_{N-1}      ↓  
                                \[W\_i^{N-1}\]  
                                   ↓  
                          \[Concatenation Split\]  
                       ↙           ↓            ↘  
                  ∂L/∂C\_{N-1} ∂L/∂B\_{N-1}  ∂L/∂I\_{N-2}  
                       ↓           ↓            ↓  
                    (continues through network)

#### Key Properties

* **Parameter Count**: O(3h²) for integration layer  
* **Flexibility**: Can learn arbitrary non-linear combinations  
* **Interpretability**: Low \- importance is distributed across weight matrix  
* **Training Dynamics**: Standard gradient flow through concatenated features

### **2.2 Learnable Integration Weights (Alpha/Beta/Gamma)**

This approach uses learnable scalar parameters to control feature mixing, providing interpretability and efficiency. The integration weights directly indicate pathway importance and adapt during training.

### **2.2.1 Direct Mixing**

#### **Design Overview**

Uses learnable scalar parameters to control feature mixing without transforming integrated features. This is the simplest and most interpretable approach.

#### **Forward Pass Visualization**

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    ↓                        ↓  
\[W\_c¹\]                    \[W\_b¹\]  
    ↓                        ↓  
Color₁                   Brightness₁  
    ↓                        ↓  
    ×α₁                     ×β₁  
    ↓                        ↓  
    └───────────┬────────────┘  
                ↓  
           Integrated₁  
                ↓  
    ┌───────────┴────────────┐  
    ↓                        ↓  
\[W\_c²\]                    \[W\_b²\]  
    ↓                        ↓  
Color₂                   Brightness₂  
    ↓                        ↓          ↓  
    ×α₂                     ×β₂        ×γ₂  
    ↓                        ↓          ↓  
    └───────────┬────────────┴──────────┘  
                ↓  
           Integrated₂  
                ↓  
    ┌───────────┴────────────┐  
    ↓                        ↓  
\[W\_c³\]                    \[W\_b³\]  
    ↓                        ↓  
Color₃                   Brightness₃  
    ↓                        ↓          ↓  
    ×α₃                     ×β₃        ×γ₃  
    ↓                        ↓          ↓  
    └───────────┬────────────┴──────────┘  
                ↓  
           Integrated₃  
                ↓  
           Classifier  
                ↓  
           Predictions

C\_l \= ReLU(W\_c^l · C\_{l-1} \+ b\_c^l)  
B\_l \= ReLU(W\_b^l · B\_{l-1} \+ b\_b^l)  
I\_l \= α\_l · C\_l \+ β\_l · B\_l \+ γ\_l · I\_{l-1}

#### **Implementation**

| class DirectMixingLayer(nn.Module):    def \_\_init\_\_(self, color\_size, brightness\_size, hidden\_size):        super().\_\_init\_\_()        \# Separate pathways        self.color\_weights \= nn.Linear(color\_size, hidden\_size)        self.brightness\_weights \= nn.Linear(brightness\_size, hidden\_size)                \# Learnable integration weights        \# Xavier/He initialization for balanced pathway         self.alpha \= nn.Parameter(torch.normal(1.0, 0.1, (1,))        self.beta \= nn.Parameter(torch.normal(1.0, 0.1, (1,))        self.gamma \= nn.Parameter(torch.normal(1.0, 0.2, (1,))            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways independently        color\_output \= F.relu(self.color\_weights(color\_input))        brightness\_output \= F.relu(self.brightness\_weights(brightness\_input))                \# Direct mixing        \# apply regularization to prevent pathway collapse        alpha\_reg \= torch.clamp(self.alpha, min=0.01)        beta\_reg \= torch.clamp(self.beta, min=0.01)        gamma\_reg \= torch.clamp(self.gamma, min=0.01)        if prev\_integrated is not None:            integrated\_output \= (alpha\_reg \* color\_output \+                                beta\_reg \* brightness\_output \+                                gamma\_reg \* prev\_integrated)        else:            integrated\_output \= alpha\_reg \* color\_output \+ beta\_reg \* brightness\_output                    return color\_output, brightness\_output, integrated\_output |
| :---- |

#### Backpropagation 

                   Loss  
                     ↓  
                Classifier  
                     ↓ ∂L/∂I₃  
               Integrated₃ \= α₃×C₃ \+ β₃×B₃ \+ γ₃×I₂  
                  ↙         ↓           ↘  
         α₃×∂L/∂I₃     β₃×∂L/∂I₃     γ₃×∂L/∂I₃  
              ↓             ↓             ↓  
         \[W\_c³\]         \[W\_b³\]           I₂  
              ↓             ↓             ↓  
     ∂L/∂W\_c³ \= α₃×∂L/∂I₃×C₂ᵀ           ↓  
              ↓             ↓             ↓  
            Color₂      Brightness₂       ↓  
              ↓             ↓             ↓  
              └─────────────┴─────────────┘  
                            ↓  
                   Integrated₂ \= α₂×C₂ \+ β₂×B₂ \+ γ₂×I₁  
                  ↙         ↓           ↘  
         α₂×∂L/∂I₂     β₂×∂L/∂I₂     γ₂×∂L/∂I₂  
              ↓             ↓             ↓  
         \[W\_c²\]         \[W\_b²\]           I₁  
              ↓             ↓             ↓  
            Color₁      Brightness₁       ↓  
              ↓             ↓             ↓  
              └─────────────┴─────────────┘  
                            ↓  
                   Integrated₁ \= α₁×C₁ \+ β₁×B₁  
                  ↙                   ↘  
         α₁×∂L/∂I₁                β₁×∂L/∂I₁  
              ↓                        ↓  
         \[W\_c¹\]                    \[W\_b¹\]  
              ↓                        ↓  
         ∂L/∂W\_c¹                 ∂L/∂W\_b¹

#### **Key Insights**

1. **Gradient Scaling**: The α, β, γ parameters directly scale the gradients

   * If α \> β, color pathway receives larger gradients  
   * Small γ prevents gradient explosion from accumulated integrated features  
2. **Parameter Updates**: Integration parameters are updated based on feature correlations

   * `∂L/∂α ∝ correlation(∂L/∂I, C)`  
   * If color features correlate with reducing loss, α increases

#### **Key Properties**

* **Parameter Count**: 3 scalar parameters  
* **Interpretability**: High \- α, β, γ directly show pathway importance  
* **Training Dynamics**: Gradients scaled by integration weights  
* **Self-Organization**: Network learns optimal mixing ratios

### **2.2.2 Channel-wise Adaptive Integration**

Instead of using scalar α, β, γ values that apply to all neurons equally, channel-wise adaptation allows each neuron/channel to have its own mixing ratios. This enables the network to learn that some features should prioritize color information while others prioritize brightness.

### **Forward Pass** 

For layer *l* with hidden size *H*:

C\_l \= ReLU(W\_c^l · C\_{l-1} \+ b\_c^l)  \# Shape: \[B, H\]  
B\_l \= ReLU(W\_b^l · B\_{l-1} \+ b\_b^l)  \# Shape: \[B, H\]

For each neuron i ∈ \[0, H-1\]:  
    I\_l\[i\] \= α\_l\[i\] · C\_l\[i\] \+ β\_l\[i\] · B\_l\[i\] \+ γ\_l\[i\] · I\_{l-1}\[i\]

Where α\_l, β\_l, γ\_l ∈ ℝ^H (vectors, not scalars)

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    ↓                        ↓  
\[W\_c¹\]                    \[W\_b¹\]  
    ↓                        ↓  
Color₁                   Brightness₁  
    ↓                        ↓  
    ×α₁\[0:511\]              ×β₁\[0:511\]  
    ↓                        ↓  
    └───────────┬────────────┘  
                ↓  
           Integrated₁  
                ↓  
    ┌───────────┴────────────┐  
    ↓                        ↓  
\[W\_c²\]                    \[W\_b²\]  
    ↓                        ↓  
Color₂                   Brightness₂  
    ↓                        ↓          ↓  
    ×α₂\[0:511\]              ×β₂\[0:511\]  ×γ₂\[0:511\]  
    ↓                        ↓          ↓  
    └───────────┬────────────┴──────────┘  
                ↓

           Integrated₂

### **Implementation**

| class ChannelAdaptiveLayer(nn.Module):    def \_\_init\_\_(self, color\_size, brightness\_size, hidden\_size,                 alpha\_init=1.0, beta\_init=1.0, gamma\_init=0.2):        super().\_\_init\_\_()                \# Pathway weights (same as Direct Mixing)        self.color\_weights \= nn.Linear(color\_size, hidden\_size)        self.brightness\_weights \= nn.Linear(brightness\_size, hidden\_size)                \# Channel-wise integration weights (not scalars\!)        self.alpha \= nn.Parameter(torch.ones(hidden\_size) \* alpha\_init)        self.beta \= nn.Parameter(torch.ones(hidden\_size) \* beta\_init)        self.gamma \= nn.Parameter(torch.ones(hidden\_size) \* gamma\_init)            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways independently        color\_output \= F.relu(self.color\_weights(color\_input))        brightness\_output \= F.relu(self.brightness\_weights(brightness\_input))                \# Channel-wise mixing        if prev\_integrated is not None:            integrated\_output \= (self.alpha \* color\_output \+                                self.beta \* brightness\_output \+                                self.gamma \* prev\_integrated)        else:            integrated\_output \= self.alpha \* color\_output \+ self.beta \* brightness\_output                    return color\_output, brightness\_output, integrated\_output |
| :---- |

### **Backward Pass**

The gradient computation is similar to Direct Mixing but now each channel has its own gradient:

**Gradient w.r.t integration parameters:**

∂L/∂α\_l\[i\] \= ∑\_batch ∂L/∂I\_l\[i\] · C\_l\[i\]  
∂L/∂β\_l\[i\] \= ∑\_batch ∂L/∂I\_l\[i\] · B\_l\[i\]

∂L/∂γ\_l\[i\] \= ∑\_batch ∂L/∂I\_l\[i\] · I\_{l-1}\[i\]

**Gradient w.r.t features:**

∂L/∂C\_l\[i\] \= α\_l\[i\] · ∂L/∂I\_l\[i\]  
∂L/∂B\_l\[i\] \= β\_l\[i\] · ∂L/∂I\_l\[i\]

∂L/∂I\_{l-1}\[i\] \= γ\_l\[i\] · ∂L/∂I\_l\[i\]

### **Backpropagation Visualization**

                  Loss  
                     ↓  
                Classifier  
                     ↓ ∂L/∂I₃  
               Integrated₃  
                  ↙  ↓  ↘  
    ┌────────────┼───┼───┼────────────┐  
    ↓            ↓   ↓   ↓            ↓  
α₃\[0\]×∂L/∂I₃\[0\]  ... ... ...  γ₃\[511\]×∂L/∂I₃\[511\]  
    ↓            ↓   ↓   ↓            ↓  
   C₃\[0\]        C₃\[i\] ↓  B₃\[511\]     I₂\[511\]  
                     ↓

              (Per-channel gradients)

### **Key Properties**

* **Parameter Count**: 3×H per layer (vs 3 scalars in Direct Mixing)  
* **Flexibility**: Each feature can have different color/brightness importance  
* **Interpretability**: Still high \- can analyze which neurons prefer color vs brightness  
* **Memory**: Same activation memory as Direct Mixing  
* **Computation**: Negligible overhead (element-wise multiplication)

### 

### **2.2.3 Dynamic Input-Dependent Integration**

Instead of fixed or per-channel weights, this approach computes integration weights dynamically based on the current input features. This allows the network to adapt its mixing strategy to the content of each image.

### **Forward Pass** 

C\_l \= ReLU(W\_c^l · C\_{l-1} \+ b\_c^l)  
B\_l \= ReLU(W\_b^l · B\_{l-1} \+ b\_b^l)

\# Dynamic weight generation  
F\_l \= \[C\_l; B\_l\]  \# Concatenated features  
w\_l \= softmax(W\_g^l · ReLU(W\_g1^l · F\_l \+ b\_g1^l) \+ b\_g^l)  
α\_l \= w\_l\[0\], β\_l \= w\_l\[1\], γ\_l \= w\_l\[2\]

\# Integration (α, β, γ are now functions of input)

I\_l \= α\_l(F\_l) · C\_l \+ β\_l(F\_l) · B\_l \+ γ\_l(F\_l) · I\_{l-1}

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    ↓                        ↓  
\[W\_c¹\]                    \[W\_b¹\]  
    ↓                        ↓  
Color₁                   Brightness₁  
    ↓                        ↓  
    └──────────┬─────────────┘  
               ↓  
        \[Concat: C₁,B₁\]  
               ↓  
       \[Weight Generator\]  
               ↓  
         (α₁, β₁) dynamic  
               ↓  
    ┌──────────┴─────────────┐  
    ↓                        ↓  
    ×α₁(input)              ×β₁(input)  
    ↓                        ↓  
    └───────────┬────────────┘  
                ↓

           Integrated₁

### **Implementation**

| class DynamicIntegrationLayer(nn.Module):    def \_\_init\_\_(self, color\_size, brightness\_size, hidden\_size):        super().\_\_init\_\_()                \# Pathway weights        self.color\_weights \= nn.Linear(color\_size, hidden\_size)        self.brightness\_weights \= nn.Linear(brightness\_size, hidden\_size)                \# Dynamic weight generator network        self.weight\_generator \= nn.Sequential(            nn.Linear(hidden\_size \* 2, hidden\_size // 4),            nn.ReLU(),            nn.Linear(hidden\_size // 4, 3),  \# Outputs α, β, γ            nn.Softmax(dim=1)  \# Ensures weights sum to 1        )            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways        color\_output \= F.relu(self.color\_weights(color\_input))        brightness\_output \= F.relu(self.brightness\_weights(brightness\_input))                \# Generate dynamic weights based on current features        if prev\_integrated is not None:            features \= torch.cat(\[color\_output, brightness\_output, prev\_integrated\], dim=1)            weights \= self.weight\_generator(features)            alpha \= weights\[:, 0:1\]             beta \= weights\[:, 1:2\]            gamma \= weights\[:, 2:3\]                        integrated\_output \= (alpha \* color\_output \+                                beta \* brightness\_output \+                                gamma \* prev\_integrated)        else:            features \= torch.cat(\[color\_output, brightness\_output\], dim=1)            weights \= self.weight\_generator(features)            alpha \= weights\[:, 0:1\]            beta \= weights\[:, 1:2\]                        integrated\_output \= alpha \* color\_output \+ beta \* brightness\_output                    return color\_output, brightness\_output, integrated\_output |
| :---- |

### **Backward Pass**

The backward pass is more complex because weights depend on features:

**Gradient through integration:**

∂L/∂C\_l \= α\_l · ∂L/∂I\_l \+ ∂L/∂α\_l · ∂α\_l/∂C\_l · C\_l \+ ∂L/∂β\_l · ∂β\_l/∂C\_l · B\_l

∂L/∂B\_l \= β\_l · ∂L/∂I\_l \+ ∂L/∂α\_l · ∂α\_l/∂B\_l · C\_l \+ ∂L/∂β\_l · ∂β\_l/∂B\_l · B\_l

**Gradient through weight generator:**

∂L/∂W\_g \= ∂L/∂weights · ∂weights/∂W\_g

where ∂L/∂weights comes from how weights affect integration

### **Backpropagation Visualization**

                    Loss  
                       ↓  
                  Classifier  
                       ↓ ∂L/∂I₂  
                 Integrated₂  
                  ↙    ↓    ↘  
         ∂L/∂α₂   ∂L/∂β₂   ∂L/∂γ₂  
              ↓        ↓        ↓  
        \[Weight Generator Network\]  
         ↙            ↓            ↘  
    ∂L/∂C₂       ∂L/∂B₂       ∂L/∂I₁

    (complex)   (complex)    (simpler)

### **Key Properties**

* **Parameter Count**: \~H²/4 \+ H/4×3 for weight generator  
* **Flexibility**: Highest \- can adapt to each input  
* **Interpretability**: Medium \- need to analyze weight generator  
* **Memory**: Slightly higher (store generated weights)  
* **Computation**: Additional forward pass through weight generator

### **2.2.4 Spatial Adaptive Integration**

Different spatial regions of an image may require different mixing strategies. For example, the sky region might rely more on brightness while a colorful object relies more on color information.

### **Forward Pass** 

For spatial position (h,w):

C\_l\[h,w\] \= ReLU(Conv(C\_{l-1})\[h,w\])  
B\_l\[h,w\] \= ReLU(Conv(B\_{l-1})\[h,w\])

I\_l\[h,w\] \= α\_l\[h,w\] · C\_l\[h,w\] \+ β\_l\[h,w\] · B\_l\[h,w\] \+ γ\_l\[h,w\] · I\_{l-1}\[h,w\]

Where α\_l, β\_l, γ\_l ∈ ℝ^{H×W} (spatial maps)

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    Size: \[B,3,32,32\]    \[B,1,32,32\]  
    ↓                        ↓  
\[Conv/Flatten\]          \[Conv/Flatten\]  
    ↓                        ↓  
Color₁                   Brightness₁  
\[B,C,H,W\]               \[B,C,H,W\]  
    ↓                        ↓  
    ×α₁\[H,W\]                ×β₁\[H,W\]  
    ↓                        ↓  
    └───────────┬────────────┘  
                ↓  
           Integrated₁

            \[B,C,H,W\]

### **Implementation**

| class SpatialAdaptiveLayer(nn.Module):    def \_\_init\_\_(self, in\_channels, out\_channels, height, width,                 alpha\_init=1.0, beta\_init=1.0, gamma\_init=0.2):        super().\_\_init\_\_()                \# Convolutional pathways (preserve spatial structure)        self.color\_conv \= nn.Conv2d(in\_channels, out\_channels, 3, padding=1)        self.brightness\_conv \= nn.Conv2d(1, out\_channels, 3, padding=1)                \# Spatial integration weights        self.alpha \= nn.Parameter(torch.ones(1, 1, height, width) \* alpha\_init)        self.beta \= nn.Parameter(torch.ones(1, 1, height, width) \* beta\_init)        self.gamma \= nn.Parameter(torch.ones(1, 1, height, width) \* gamma\_init)            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways (maintaining spatial dimensions)        color\_output \= F.relu(self.color\_conv(color\_input))        brightness\_output \= F.relu(self.brightness\_conv(brightness\_input))                \# Spatial mixing        if prev\_integrated is not None:            integrated\_output \= (self.alpha \* color\_output \+                                self.beta \* brightness\_output \+                                self.gamma \* prev\_integrated)        else:            integrated\_output \= self.alpha \* color\_output \+ self.beta \* brightness\_output                    return color\_output, brightness\_output, integrated\_output |
| :---- |

### **Backward Pass**

**Gradient w.r.t spatial weights:**

∂L/∂α\_l\[h,w\] \= ∑\_{batch,channel} ∂L/∂I\_l\[b,c,h,w\] · C\_l\[b,c,h,w\]

∂L/∂β\_l\[h,w\] \= ∑\_{batch,channel} ∂L/∂I\_l\[b,c,h,w\] · B\_l\[b,c,h,w\]

**Gradient w.r.t features:**

∂L/∂C\_l\[b,c,h,w\] \= α\_l\[h,w\] · ∂L/∂I\_l\[b,c,h,w\]

∂L/∂B\_l\[b,c,h,w\] \= β\_l\[h,w\] · ∂L/∂I\_l\[b,c,h,w\]

### **Backpropagation Visualization**

                  Loss  
                     ↓  
                Classifier  
                     ↓  
               Integrated₃  
              \[B,C,H,W\]  
                     ↓  
    ┌────────────────┼────────────────┐  
    ↓                ↓                ↓  
Spatial α\[h,w\]   Spatial β\[h,w\]   Spatial γ\[h,w\]  
    ↓                ↓                ↓  
   C₃\[h,w\]         B₃\[h,w\]         I₂\[h,w\]  
    

Each spatial location has independent gradients

### **Key Properties**

* **Parameter Count**: 3×H×W per layer (can be significant for large feature maps)  
* **Flexibility**: Different mixing per spatial location  
* **Interpretability**: Very high \- can visualize spatial attention maps  
* **Memory**: Same as convolutional networks  
* **Computation**: Element-wise multiplication with spatial broadcasting

### **2.3 Mixing With Neural Processing**

#### **Design Overview**

Applies neural transformation to integrated features before mixing, providing a balance between flexibility and interpretability.

#### **Forward Pass Visualization**

Input (RGB+L)  
    ↓  
Split: \[RGB\] → Color₀, \[L\] → Brightness₀  
    ↓                        ↓  
\[W\_c¹\]                    \[W\_b¹\]  
    ↓                        ↓  
Color₁                   Brightness₁  
    ↓                        ↓  
    ×α₁                     ×β₁  
    ↓                        ↓  
    └───────────┬────────────┘  
                ↓  
           Integrated₁  
                ↓  
    ┌───────────┼────────────┐  
    ↓           ↓            ↓  
\[W\_c²\]      \[W\_b²\]        \[W\_i²\]  
    ↓           ↓            ↓  
Color₂    Brightness₂    Ĩ₂(processed)  
    ↓           ↓            ↓  
    ×α₂        ×β₂          ×γ₂  
    ↓           ↓            ↓  
    └───────────┴────────────┘  
                ↓  
           Integrated₂  
                ↓  
    ┌───────────┼────────────┐  
    ↓           ↓            ↓  
\[W\_c³\]      \[W\_b³\]        \[W\_i³\]  
    ↓           ↓            ↓  
Color₃    Brightness₃    Ĩ₃(processed)  
    ↓           ↓            ↓  
    ×α₃        ×β₃          ×γ₃  
    ↓           ↓            ↓  
    └───────────┴────────────┘  
                ↓  
           Integrated₃  
                ↓  
           Classifier  
                ↓  
           Predictions

C\_l \= ReLU(W\_c^l · C\_{l-1} \+ b\_c^l)  
B\_l \= ReLU(W\_b^l · B\_{l-1} \+ b\_b^l)  
Ĩ\_l \= ReLU(W\_i^l · I\_{l-1} \+ b\_i^l)  
I\_l \= α\_l · C\_l \+ β\_l · B\_l \+ γ\_l · Ĩ\_l

Where `Ĩ_l` represents the processed integrated features.

#### **Implementation**

| class NeuralProcessingLayer(nn.Module):    def \_\_init\_\_(self, color\_size, brightness\_size, hidden\_size):        super().\_\_init\_\_()        \# Separate pathways        self.color\_weights \= nn.Linear(color\_size, hidden\_size)        self.brightness\_weights \= nn.Linear(brightness\_size, hidden\_size)                \# Integrated pathway transformation        self.integrated\_weights \= nn.Linear(hidden\_size, hidden\_size)                \# Learnable mixing weights        self.alpha \= nn.Parameter(torch.normal(1.0, 0.1, (1,)))        self.beta \= nn.Parameter(torch.normal(1.0, 0.1, (1,)))        self.gamma \= nn.Parameter(torch.normal(0.1, 0.02, (1,)))            def forward(self, color\_input, brightness\_input, prev\_integrated=None):        \# Process pathways independently        color\_output \= F.relu(self.color\_weights(color\_input))        brightness\_output \= F.relu(self.brightness\_weights(brightness\_input))                \# Apply regularization to prevent pathway collapse        alpha\_reg \= torch.clamp(self.alpha, min=0.01)        beta\_reg \= torch.clamp(self.beta, min=0.01)        gamma\_reg \= torch.clamp(self.gamma, min=0.01)        \# Neural processing of integrated        if prev\_integrated is not None:            integrated\_processed \= F.relu(self.integrated\_weights(prev\_integrated))            integrated\_output \= (alpha\_reg \* color\_output \+                                beta\_reg \* brightness\_output \+                                gamma\_reg \* integrated\_processed)        else:            integrated\_output \= alpha\_reg \* color\_output \+ beta\_reg \* brightness\_output                    return color\_output, brightness\_output, integrated\_output |
| :---- |

#### Backpropagation Visualization

                   Loss  
                     ↓  
                Classifier  
                     ↓ ∂L/∂I₃  
               Integrated₃ \= α₃×C₃ \+ β₃×B₃ \+ γ₃×Ĩ₃  
                  ↙         ↓           ↘  
         α₃×∂L/∂I₃     β₃×∂L/∂I₃     γ₃×∂L/∂I₃  
              ↓             ↓             ↓  
            Color₃    Brightness₃        Ĩ₃  
              ↓             ↓             ↓  
         \[W\_c³\]         \[W\_b³\]        \[W\_i³\]  
              ↓             ↓             ↓  
     ∂L/∂W\_c³ \=     ∂L/∂W\_b³ \=    ∂L/∂W\_i³ \=   
     α₃×∂L/∂I₃×C₂ᵀ  β₃×∂L/∂I₃×B₂ᵀ  γ₃×∂L/∂I₃×I₂ᵀ  
              ↓             ↓             ↓  
            Color₂    Brightness₂    ∂L/∂I₂ \= γ₃×∂L/∂I₃×W\_i³ᵀ  
              ↓             ↓             ↓  
              └─────────────┴─────────────┘  
                            ↓  
                   Integrated₂ \= α₂×C₂ \+ β₂×B₂ \+ γ₂×Ĩ₂  
                  ↙         ↓           ↘  
         α₂×∂L/∂I₂     β₂×∂L/∂I₂     γ₂×∂L/∂I₂  
              ↓             ↓             ↓  
            Color₂    Brightness₂        Ĩ₂  
              ↓             ↓             ↓  
         \[W\_c²\]         \[W\_b²\]        \[W\_i²\]  
              ↓             ↓             ↓  
            Color₁    Brightness₁    ∂L/∂I₁ \= γ₂×∂L/∂I₂×W\_i²ᵀ  
              ↓             ↓             ↓  
              └─────────────┴─────────────┘  
                            ↓  
                   Integrated₁ \= α₁×C₁ \+ β₁×B₁  
                  ↙                   ↘  
         α₁×∂L/∂I₁                β₁×∂L/∂I₁  
              ↓                        ↓  
         \[W\_c¹\]                    \[W\_b¹\]  
              ↓                        ↓  
         ∂L/∂W\_c¹                 ∂L/∂W\_b¹

#### **Key Properties**

* **Parameter Count**: h² \+ 3 scalars  
* **Flexibility**: Medium \- can transform integrated features  
* **Interpretability**: Medium \- α, β, γ show importance, but transformation is opaque  
* **Use Case**: When integrated features need refinement between layers

### Further Thoughts: Attention-Based Multi-Stream Neurons

This approach uses multiple weight matrices to implement learnable cross-modal attention:

**Implementation:**

| class AttentionMultiWeightNN:    def \_\_init\_\_(self, input\_size):        \# Direct processing weights        self.color\_weights \= initialize\_weights(input\_size)        self.brightness\_weights \= initialize\_weights(input\_size)                \# Cross-modal attention weights        self.color\_to\_brightness\_weights \= initialize\_weights(input\_size)        self.brightness\_to\_color\_weights \= initialize\_weights(input\_size)                \# Separate biases        self.color\_bias \= initialize\_bias()        self.brightness\_bias \= initialize\_bias()            def forward(self, inputs):        color\_components \= extract\_color\_features(inputs)        brightness\_components \= extract\_brightness\_features(inputs)                \# Direct processing paths        color\_direct \= dot(self.color\_weights, color\_components)        brightness\_direct \= dot(self.brightness\_weights, brightness\_components)                \# Cross-modal attention computation        brightness\_to\_color\_attention \= dot(self.brightness\_to\_color\_weights, brightness\_components)        color\_to\_brightness\_attention \= dot(self.color\_to\_brightness\_weights, color\_components)                \# Separate outputs with cross-modal influence        color\_output \= activation\_fn(color\_direct \+ brightness\_to\_color\_attention \+ self.color\_bias)        brightness\_output \= activation\_fn(brightness\_direct \+ color\_to\_brightness\_attention \+ self.brightness\_bias)                return color\_output, brightness\_output |
| :---- |

This maintains feature separation while enabling learned cross-modal interactions, making it functionally similar to Option 1C but with more sophisticated attention mechanisms.

# **Dataset Guide for Multi-Stream Neural Network Research**

To thoroughly test and validate our Multi-Stream neural network approach, we'll use a phased approach with increasingly specialized datasets \- from derived color/brightness data to true multimodal sensor data.

## **Initial Testing with Derived Data \- RGB+Luminance** 

We'll begin with datasets where color and brightness information can be separated through transformation:

### **Standard Computer Vision Datasets**

For scaling up experiments with derived data:

* **ImageNet**: Large-scale dataset with over a million images  
* **CIFAR-10**: 60,000 32×32 color images in 10 classes  
* **COCO**: 328,000 images with object detection and segmentation annotations

These standard datasets can be preprocessed to separate color and brightness information through color space transformations.

### **Input Data Representation**

Our implementation preserves all original RGB information while explicitly adding brightness data as a 4th channel. For each image:

* **Channels 1-3**: Original RGB data (unchanged)  
* **Channel 4**: Computed luminance using ITU-R BT.709 standard weights (appropriate for sRGB color space):  
  * L \= 0.2126×R \+ 0.7152×G \+ 0.0722×B

This approach ensures zero information loss while providing explicit brightness information for specialized processing. Note: these coefficients are correct for standard digital images (sRGB) which encompass most computer vision datasets.

### **Architecture Design**

The network maintains separate processing pathways from the input layer:

1. **Color Pathway**: Processes RGB channels (channels 1-3) through specialized color weights (wc)  
2. **Brightness Pathway**: Processes luminance channel (channel 4\) through specialized brightness weights (wb)  
3. **Integration Module**: Implements continuous learnable integration with trainable weights that determine optimal combination of color and brightness features

### **Key Advantages**

* **No Information Loss**: Original RGB data is preserved completely, avoiding quantization errors from color space conversions  
* **Clean Separation**: Color and brightness information are separated from the input layer, preventing feature mixing in early layers  
* **Biological Alignment**: Mirrors the human visual system's separate processing of color (parvocellular) and brightness (magnocellular) pathways  
* **Implementation Simplicity**: Requires only adding one channel to standard RGB inputs, making it compatible with existing ImageNet preprocessing pipelines

### **Implementation Formula**

For a batch of images with shape (B, 3, H, W), the transformation produces:

* **Input**: (B, 3, H, W) → **Output**: (B, 4, H, W)  
* Memory overhead: 33% increase (from 3 to 4 channels)  
* Computation: One additional weighted sum per pixel

This design enables the network to learn specialized representations for color and brightness while maintaining the flexibility to integrate these features optimally for each task.

## 

## **Custom Multimodal Luminance Datasets**

For definitive validation, we'll need to create custom datasets with true separate luminance measurements:

### **RGB Camera \+ Light Meter Setup**

* Pair a standard RGB camera with calibrated lux/luminance meters  
* Capture simultaneous RGB images and precise brightness measurements  
* Sample a range of environments with varying lighting conditions  
* Create ground truth data for both color-based and brightness-based classification

### **RGB-NIR Camera Systems**

* Use specialized cameras that capture both visible light (RGB) and near-infrared (NIR)  
* NIR channel provides additional luminance information independent from color  
* Creates true multimodal data from physical sensors

### **Smartphone Sensor Fusion**

* Modern smartphones have both cameras and dedicated ambient light sensors  
* Custom app to collect paired RGB images and lux readings  
* Enables collection of large datasets with minimal equipment

## 

## **Evaluation Strategy**

We'll use a consistent evaluation methodology across all three phases:

1. **Baseline**: Train standard neural networks on RGB inputs  
2. **Preprocessed Baseline**: Train standard networks on transformed inputs  
3. **Multi-Stream Network**: Train our architecture with separate color and brightness weights

For each phase, we'll systematically evaluate:

* Overall classification accuracy  
* Performance under varying lighting conditions  
* Performance with color variations  
* Robustness to noise in either color or brightness data

This progressive approach allows us to:

1. Quickly test and refine our architecture with readily available data  
2. Demonstrate increasing improvements as we move to true multimodal data  
3. Provide compelling evidence that Multi-Stream networks have advantages over traditional architectures

### **Baseline Comparisons**

To properly evaluate the effectiveness of Multi-Stream approaches, we will compare against:

1. **Standard RGB Network**: Traditional neural network using only RGB input data  
2. **Dual Network Ensemble**: Two separate networks \- one trained on color data and one on brightness data \- with outputs combined at the end  
3. **Preprocessed Single Network**: Traditional network using transformed inputs

This comparison framework will demonstrate whether the Multi-Stream architecture provides advantages over both simple single-modal approaches and traditional ensemble methods.

### **Evaluation metrics**

1. **Classification Accuracy**: Standard benchmarks like ImageNet  
2. **Computational efficiency:** Training time, inference speed comparisons, and   
3. **Adversarial Robustness**: Specifically against color and brightness perturbations  
4. **Lighting Invariance**: Performance across images with different lighting conditions  
5. **Cross-Dataset Generalization**: Performance when trained on one dataset and tested on another  
6. **Robustness to lighting changes**: synthetic data with varied illumination  
7. **Color deficient vision simulation: performance under simulated color blindness conditions**

## **Expected Benefits**

1. **Improved Recognition Performance**: Separating processing pathways should allow for more refined feature extraction  
2. **Robustness to Lighting Conditions**: By separating brightness from color, the network may better handle varying illumination  
3. **Better Feature Integration**: The network can learn when to prioritize color vs. brightness depending on the recognition task  
4. **Closer Alignment with Human Perception**: May produce results more consistent with human visual perception

## **Technical Challenges**

1. **Increased Parameter Count**: Multiple weights increase model complexity  
2. **Feature Separation**: Properly separating color and brightness information  
3. **Integration Mechanism**: Finding optimal ways to combine information from different weight channels  
4. **Training Stability**: Ensuring balanced learning across different weight types  
5. **Gradient flow management:** monitoring gradient magnitudes across pathways to prevent vanishing/exploding gradients  
6. **Memory efficiency:** managing increased memory requirements during training 

## **Conclusion**

Multi-Stream Neural Networks represent a promising approach to improving computer vision systems by more closely mimicking the specialized processing pathways found in biological visual systems. The proposed architecture offers a principled way to separate and integrate different aspects of visual information, potentially leading to more robust and efficient image recognition systems.

The phased evaluation approach, progressing from derived data to true multimodal sensor inputs, provides a comprehensive framework for validating the effectiveness of this architecture. Early results with derived data will inform the design of more sophisticated implementations using genuine multimodal sensor data.

[image1]: <data:image/png;base64,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>