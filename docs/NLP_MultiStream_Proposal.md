# Multi-Stream Neural Networks for NLP: Research Proposal

## Overview

This proposal outlines an extension of our biologically-inspired multi-stream neural network architecture from computer vision to natural language processing (NLP). Just as our vision models integrate complementary sensory modalities (RGB + Depth) to improve scene understanding, this NLP approach will integrate complementary linguistic representations (Semantic + Phonetic + Morphological) for enhanced text classification.

---

## 1. Motivation and Research Question

### Biological Integration Principle

**Dendritic Integration Model**: The core architectural principle is inspired by how biological neurons integrate information. Real neurons integrate **graded potentials from dendritic filtering** at the soma **before** applying a firing threshold—not after.

**Our design mirrors this**: Each stream (Semantic, Phonetic, Morphological) performs dendritic filtering on its input, then these raw filtered outputs are integrated using learnable weights, and finally a threshold (ReLU) is applied to the integrated result.

**Why this matters**: Standard post-activation fusion combines already-thresholded signals (like combining action potentials). Our architecture instead **continuously integrates streams at the neuron level**—just as biological neurons continuously combine dendritic inputs—enabling the network to discover optimal combinations of complementary features.

### Vision Analogy
**Current CV Work:**
- **Sensors**: RGB camera (color) + Depth sensor (distance)
- **Task**: Scene classification (SUN-RGBD dataset)
- **Architecture**: ResNet backbone with multi-stream integration (LINet3, DMNet)
- **Key insight**: Different sensors provide complementary information about the same scene

### NLP Extension
**Proposed NLP Work:**
- **"Sensors"**: Semantic (meaning) + Phonetic (sound) + Morphological (structure)
- **Task**: Text classification (sentiment, topic, etc.)
- **Architecture**: BERT backbone with multi-stream integration (LI-BERT, DM-BERT)
- **Key insight**: Different linguistic representations provide complementary information about the same text

### Research Question
**Can biologically-inspired multi-stream neural architectures that integrate semantic, phonetic, and morphological representations improve text classification performance compared to single-stream transformer models?**

---

## 2. Multi-Stream Input Modalities

### Stream 1: Semantic/Lexical
**What it captures**: Word meanings, semantic similarity, distributional semantics

**Encoding method**:
- Standard word-level tokenization (BERT tokenizer)
- Pretrained word embeddings or contextual embeddings

**Biological parallel**: Wernicke's area (semantic processing)

**Example**:
```
Text: "This movie was unbelievably good!"
Semantic tokens: ["this", "movie", "was", "unbelievably", "good"]
```

---

### Stream 2: Phonetic/Prosodic
**What it captures**: Sound patterns, pronunciation, prosodic features, emotional tone

**Encoding method**:
- Grapheme-to-phoneme (G2P) conversion using tools like `g2p-en`
- Phoneme-level embeddings
- Stress and syllable patterns

**Biological parallel**: Superior temporal gyrus (auditory/phonetic processing)

**Example**:
```
Text: "This movie was unbelievably good!"
Phonetic: [ðɪs], [ˈmuːvi], [wɒz], [ˌʌnbɪˈliːvəbli], [ɡʊd]
Key features: Emphatic stress on "un-be-LIEV-ably" → intensification
```

**Why it matters**:
- Captures prosodic emphasis (intensifiers, sarcasm)
- Detects phonetic misspellings in informal text
- Provides acoustic-like features for written language

---

### Stream 3: Morphological/Syntactic
**What it captures**: Word formation, grammatical structure, compositional meaning

**Encoding method**:
- Morpheme decomposition using spaCy
- Part-of-speech (POS) tags
- Dependency relations
- Morphological features (tense, number, etc.)

**Biological parallel**: Broca's area (syntactic/grammatical processing)

**Example**:
```
Text: "This movie was unbelievably good!"
Morphological: ["this_DET", "movie_NOUN", "be_AUX+PAST", "un+believe+able+ly_ADV", "good_ADJ"]
Key features: "un-" prefix + "-ably" suffix → paradoxical intensification despite negation
```

**Why it matters**:
- Captures compositional semantics (prefixes, suffixes modify meaning)
- Detects grammatical patterns (negation, intensification)
- Handles morphologically-rich languages better

---

## 3. Integration Architecture

### Approach 1: Linear Integration (LI-BERT)
- Full **linear projection** weight matrices for stream integration (NLP equivalent of Conv1x1)
- Layer normalization on integrated stream
- More parameters, potentially better performance
- **Biological analogy**: Rich synaptic connectivity between linguistic processing areas

### Approach 2: Direct Mixing (DM-BERT)
- Scalar mixing weights for stream integration
- Linear projection on recurrent connection (integrated_prev)
- NO layer normalization on integrated stream (raw "membrane potential")
- Fewer parameters, more biologically plausible
- **Biological analogy**: Simple weighted summation of dendritic inputs, minimal normalization

### Key Features (from CV architecture)
- **Raw integration**: Integrate streams WITHOUT their biases (soma integration principle)
- **Recurrent integration**: Previous integrated state informs current integration
- **Stream independence**: Each stream processes its representation independently before integration

---

## 4. Use Case: Text Classification

### Primary Task
**Sentiment Analysis** (fine-grained)
- Binary: Positive/Negative
- Fine-grained: Very Negative, Negative, Neutral, Positive, Very Positive

### Why Sentiment?
1. **Complementary streams matter**:
   - Semantic: Literal word meanings
   - Phonetic: Emotional prosody, emphasis patterns
   - Morphological: Intensifiers (un-, very-, -ly), negation handling

2. **Clear evaluation**: Standard benchmarks with published baselines

3. **Practical applications**: Product reviews, social media analysis, customer feedback

### Example Benefit
```
Text: "This movie was unbelievably good!"

Single-stream (semantic only):
- "good" = positive (maybe neutral without context)

Multi-stream integration:
- Semantic: "good" = positive
- Phonetic: Emphatic stress on "unbelievably" = strong emotion
- Morphological: "un-" paradoxically intensifies (not negates) + "-ly" adverb
- Integrated: STRONG positive sentiment ✓
```

---

## 5. Baseline Models and Comparisons

### Model Hierarchy (like CV ResNet experiments)

| CV Architecture | NLP Equivalent | Description |
|----------------|----------------|-------------|
| ResNet-18 (RGB only) | **BERT-base (semantic only)** | Single-stream baseline |
| ResNet-18 (RGB+D early fusion) | **BERT (concat all streams)** | Naive concatenation baseline |
| ResNet-18 (RGB+D late fusion) | **BERT (ensemble)** | Train separately, combine outputs |
| **LINet3 (RGB+D)** | **LI-BERT (semantic+phonetic+morphological)** | Linear Integration (proposed) |
| **DMNet (RGB+D)** | **DM-BERT (semantic+phonetic+morphological)** | Direct Mixing (proposed) |

### Published Baselines to Compare

**BERT and Variants:**
- **BERT-base**: 110M parameters, standard baseline
- **RoBERTa**: Improved BERT training, better baseline
- **ELECTRA-small**: Efficient discriminative pre-training
- **DistilBERT**: Lightweight BERT (66M params)

**Multi-task/Multi-modal NLP:**
- **LISA** (Strubell et al., 2018): Syntax-aware multi-task learning
- **Phonologically-informed LMs** (Choi et al., 2020): Phonetic + semantic streams

---

## 6. Datasets

### Primary Dataset: Stanford Sentiment Treebank (SST-5)
**Why SST-5 is our "SUN-RGBD equivalent":**

| Characteristic | SUN-RGBD (CV) | SST-5 (NLP) |
|----------------|---------------|-------------|
| **Task** | Scene classification | Fine-grained sentiment |
| **Size** | ~10K images | ~11.8K sentences |
| **Modalities** | RGB + Depth | Text (generates 3 streams) |
| **Classes** | 19 scene categories | 5 sentiment levels |
| **Established** | ✓ Standard benchmark | ✓ Standard benchmark |

**Dataset Details:**
- **Size**: 11,855 sentences
- **Classes**: Very Negative, Negative, Neutral, Positive, Very Positive
- **Domain**: Movie reviews (rich linguistic variation)
- **Published baseline**: BERT-base achieves ~52.3% accuracy (5-way)
- **Human performance**: ~80.7% accuracy

**Why good for multi-stream:**
- Emotional language benefits from phonetic/prosodic cues
- Sentiment modifiers (un-, -ly, very-, etc.) captured by morphology
- Fine-grained distinctions require nuanced understanding

---

### Alternative/Extension Datasets

| Dataset | Task | Size | Why Good | BERT Baseline |
|---------|------|------|----------|---------------|
| **SST-2** | Binary sentiment | 67K sentences | Simpler, faster experiments | 93.5% |
| **IMDb** | Binary sentiment | 50K reviews | Longer documents, emotional prosody | 94.5% |
| **AG News** | Topic classification (4-way) | 120K articles | Morphology helps with technical terms | 94.6% |
| **TREC** | Question classification | 5.5K questions | Syntax matters (wh-words) | 98% |
| **Hate Speech** | Hate/offensive/neutral | 25K tweets | Phonetic obfuscation, morphological creativity | 90% |

---

## 7. Data Preprocessing

### Automatic Multi-Stream Generation
**Key advantage**: All three streams derived from single text source (no translation needed)

```
Input text: "This movie was unbelievably good!"

Stream 1 (Semantic):
  Tool: BERT tokenizer
  Output: ["this", "movie", "was", "unbelievably", "good"]

Stream 2 (Phonetic):
  Tool: g2p-en (Grapheme-to-Phoneme)
  Output: ["DH", "IH", "S", "M", "UW", "V", "IY", ...]

Stream 3 (Morphological):
  Tool: spaCy (morphological analysis)
  Output: ["this_DET", "movie_NOUN", "be_AUX+PAST", "un+believe+able+ly_ADV", "good_ADJ"]
```

### Tools and Libraries
- **Semantic**: HuggingFace Transformers (BERT tokenizer)
- **Phonetic**: `g2p-en` or `epitran` for G2P conversion
- **Morphological**: spaCy (`en_core_web_sm`) for POS + morphology

---

## 8. Evaluation Metrics

### Primary Metrics
- **Accuracy**: Standard classification accuracy (match CV experiments)
- **F1-score**: Macro-averaged F1 (handles class imbalance)

### Analysis Metrics (from CV work)
- **Stream contribution analysis**: Which stream dominates? (α weights for DM-BERT)
- **Ablation studies**: Performance without each stream
- **Error analysis**: Where does multi-stream help most?

### Comparison Points
1. **vs. Single-stream BERT**: Does multi-stream improve over semantic-only?
2. **vs. Early fusion**: Does architectural integration beat naive concatenation?
3. **vs. Late fusion**: Does joint training beat separate models?
4. **LI-BERT vs. DM-BERT**: Does biological plausibility trade off with performance?

---

## 9. Expected Contributions

### Novel Contributions
1. **First application** of biologically-inspired multi-stream integration (LI/DM architectures) to NLP
2. **Systematic comparison** of semantic, phonetic, and morphological streams for text classification
3. **Analysis** of learned stream mixing weights (which linguistic representation matters most?)
4. **Architectural insights** from CV→NLP transfer (do biological principles generalize?)

### Positioning vs. Prior Work
- **vs. Multi-task learning**: True multi-stream architecture (not auxiliary losses)
- **vs. Ensemble methods**: Joint training with architectural integration (not post-hoc combination)
- **vs. Feature concatenation**: Learned integration weights (not fixed combination)
- **vs. Phonetic LMs**: Addition of morphological stream + biologically-inspired integration

---

## 10. Experimental Timeline

### Phase 1: Baseline Implementation (2 weeks)
- Set up SST-5 dataset with multi-stream preprocessing
- Implement BERT-base baseline (semantic only)
- Verify reproduction of published results (~52.3%)

### Phase 2: Multi-Stream Architectures (2 weeks)
- Implement LI-BERT (Linear Integration)
- Implement DM-BERT (Direct Mixing)
- Implement comparison baselines (early fusion, late fusion)

### Phase 3: Training and Evaluation (2 weeks)
- Train all models on SST-5
- Hyperparameter tuning
- Ablation studies (remove each stream)

### Phase 4: Analysis and Extension (2 weeks)
- Analyze stream mixing weights
- Error analysis
- Test on alternative datasets (SST-2, IMDb)

---

## 11. Success Criteria

### Minimum Success
- Multi-stream model (LI-BERT or DM-BERT) outperforms BERT-base on SST-5
- Ablation shows each stream contributes unique information
- Analysis reveals interpretable stream mixing patterns

### Strong Success
- Multi-stream model achieves new state-of-the-art on SST-5
- Consistent improvement across multiple datasets (SST-2, IMDb, AG News)
- Clear mapping between stream weights and linguistic phenomena

### Stretch Goals
- Demonstrate transfer to other languages (multilingual BERT backbone)
- Show robustness to adversarial examples (phonetic/morphological attacks)
- Publish findings at NLP conference (ACL, EMNLP, NAACL)

---

## 12. Potential Challenges and Mitigation

### Challenge 1: Data Preprocessing Complexity
**Issue**: Generating phonetic and morphological streams requires additional tools
**Mitigation**: Use well-established libraries (g2p-en, spaCy), cache preprocessed data

### Challenge 2: Alignment Between Streams
**Issue**: Different tokenization lengths (words vs. phonemes vs. morphemes)
**Mitigation**: Use attention-based pooling, positional encodings, or character-level alignment

### Challenge 3: Computational Cost
**Issue**: Three encoders instead of one increases training time
**Mitigation**: Use smaller encoders for phonetic/morphological streams, efficient architectures (ELECTRA-small)

### Challenge 4: Limited Improvement
**Issue**: Multi-stream may not significantly outperform strong BERT baseline
**Mitigation**: Focus on interpretability, specific linguistic phenomena, low-resource settings

---

## 13. Related Work

### Multi-Stream NLP
- **Strubell et al. (2018)**: "Linguistically-Informed Self-Attention" - syntax-aware BERT
- **Choi et al. (2020)**: "Phonologically and Semantically Informed LMs" - phonetic features for language modeling

### Multi-Modal Learning
- **Lu et al. (2019)**: "ViLBERT" - vision + language pre-training
- **Tan & Bansal (2019)**: "LXMERT" - cross-modal transformer

### Biologically-Inspired Architectures
- **Our CV work**: LINet3, DMNet for RGB-D scene classification
- **Spoerer et al. (2017)**: Recurrent CNNs inspired by visual cortex
- **Lindsay & Miller (2018)**: Attention as routing in biological neural networks

---

## 14. Future Directions

### Short-term Extensions
- **More streams**: Add pragmatic features (speaker identity, discourse markers)
- **More tasks**: Question answering, natural language inference, text generation
- **More languages**: Multilingual BERT backbone for cross-lingual transfer

### Long-term Vision
- **Unified multi-stream framework**: Single architecture for vision + language (CLIP-style)
- **Adaptive stream selection**: Learn which streams matter for which examples
- **Neuroscience validation**: Compare learned representations to brain activation patterns (fMRI)

---

## References

### Our Prior Work
- Multi-stream neural networks for RGB-D scene classification (SUN-RGBD)
- Linear Integration (LINet3) and Direct Mixing (DMNet) architectures
- Biologically-inspired integration without batch normalization on integrated stream

### Key NLP Baselines
- BERT: Devlin et al. (2019)
- RoBERTa: Liu et al. (2019)
- SST-5: Socher et al. (2013)

### Multi-Stream NLP
- LISA: Strubell et al. (2018)
- Phonological LMs: Choi et al. (2020)

---

## Appendix: Architecture Details

### LI-BERT (Linear Integration)
```
For each transformer layer:
  1. Process semantic, phonetic, morphological streams independently
  2. Integrate using Conv1x1 weights:
     integrated = W_prev·prev + Σ(W_i·stream_i_raw) + bias
  3. Apply batch normalization to integrated stream
  4. Apply ReLU activation
  5. Pass to next layer
```

### DM-BERT (Direct Mixing)
```
For each transformer layer:
  1. Process semantic, phonetic, morphological streams independently
  2. Integrate using scalar weights:
     integrated = γ·Conv1x1(prev) + Σ(α_i·stream_i_raw) + bias
  3. NO batch normalization (raw membrane potential)
  4. Apply ReLU activation
  5. Pass to next layer
```

### Key Difference from Standard BERT
- **Standard BERT**: Single semantic stream, self-attention within stream
- **LI/DM-BERT**: Three streams, cross-stream integration at each layer, learned mixing weights

---

**Document Version**: 1.0
**Date**: 2024
**Status**: Research Proposal (Pre-Implementation)
