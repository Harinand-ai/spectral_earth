# Controlled Three-Way Active Learning Comparison: Methodology

## 1. Objective and Research Hypothesis
The primary objective of ConformAL-HSI is label-efficient hyperspectral classification. This study directly evaluates whether our updated active learning strategy (**Margin Sampling + Cosine Diversity**) delivers measurable label-efficiency improvements over our original strategy (**Least Confidence + Farthest-First Diversity**) and baseline **Random Sampling** under strict identical experimental controls.

## 2. Experimental Controls
- **Classifier Architecture:** Original ClassifierHead:
  $$\text{Linear}(2048 \to 256) \to \text{LayerNorm}(256) \to \text{GELU}() \to \text{Linear}(256 \to C)$$
  Preceded by representation $\text{LayerNorm}(2048)$ without dropout (~0.53M trainable parameters).
- **Backbone Embeddings:** Pretrained SpectralEarth 3D-ResNet50 frozen embeddings ($D=2048$), with unit $L_2$ normalization for Cosine metrics.
- **Initial Training Set:** Exactly 10 samples per class drawn using paired random seeds, shared identically across all three methods.
- **Acquisition Batch Size:** Fixed batch size $k = 20$ samples per acquisition round.
- **Progression:** 10 consecutive acquisition rounds evaluated on strictly held-out test sets.
- **Seeds:** 5 independent random seeds ($42, 43, 44, 45, 46$).
- **Optimizer:** AdamW ($\eta = 0.0005$, weight decay $\lambda = 10^{-4}$), Cosine Annealing over 25 epochs per round, batch size 64.

## 3. Method Implementations
1. **Random Sampling:** Uniform random selection of $k=20$ samples from the available unlabeled training pool.
2. **Original LC + FD:** 
   - Least Confidence uncertainty: $u_i = 1 - \max_c p(y=c|x_i)$.
   - Candidate filtering: top $M = \min(|U|, \max(4k, 50))$ samples.
   - Euclidean Farthest-First Traversal on raw embeddings within candidates.
3. **Updated Margin + Cosine:**
   - Margin uncertainty: $m_i = 1 - (p_{(1)} - p_{(2)})$.
   - Candidate filtering: top $M = \min(|U|, \max(4k, 200))$ samples.
   - Cosine Farthest-First Traversal relative to the existing labeled pool $L$.
