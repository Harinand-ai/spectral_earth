# FINAL METHODOLOGY — SPECTRALEARTH ACTIVE LEARNING FRAMEWORK

## 1. Architectural Overview

The **SpectralEarth Hyperspectral Active Learning Framework** decouples representation learning, classifier training, sample selection, and calibration into two modular engines:

```
Input Hyperspectral Patch x ∈ R^{B × H × W}
             ↓
    [100% Frozen Backbone]
SpectralEarth spec_resnet50 (24.2M params)
             ↓
2048-dimensional Feature Embedding φ(x) ∈ R^{2048}
             ↓
   [Trainable Classifier Head]
  MLP-2Layer + LayerNorm + GELU (533K params)
             ↓
   Uncalibrated Logits z(x) ∈ R^C
             ↓
   [Temperature Scaler]  ←  Fitted ONLY on Validation Set (D_val)
   Calibrated Probabilities p(x) = softmax(z(x)/T)
             ↓
  ┌──────────┴──────────┐
  │  Model B Engine     │
  │  - Calibrated LC    │
  │  - Top-M Filtering  │
  │  - Labeled Diversity│
  └──────────┬──────────┘
             ↓
  Selected k Queries → Oracle Labeling → Update L & U Pools → Retrain Head
```

---

## 2. Mathematical Formulations

### A. Feature Extraction & Caching
Let $x_i$ be a spatial-spectral hyperspectral patch of size $B \times H \times W$, where $B$ is the number of spectral channels.
The feature embedding $\phi(x_i) \in \mathbb{R}^{2048}$ is computed by the frozen SpectralEarth backbone $f_{\theta}$:
$$\phi(x_i) = f_{\theta}(x_i), \quad \nabla_{\theta} = 0$$
Because $\theta$ is strictly frozen across all AL rounds, $\phi(x_i)$ is computed once and stored in memory:
$$\mathcal{C} = \{ (\phi(x_i), y_i) \}_{i=1}^N$$

### B. Trainable Classifier Head & Loss Functions
The classifier head $g_{\psi}$ maps 2048-dimensional feature vectors to class logits:
$$z_i = g_{\psi}(\phi(x_i)) = W_2 \cdot \text{GELU}(\text{LayerNorm}(W_1 \phi(x_i) + b_1)) + b_2$$
where $\psi = \{W_1, b_1, W_2, b_2\}$ are trainable parameters (533,264 parameters).

**Standard Cross-Entropy Loss**:
$$\mathcal{L}_{\text{CE}}(\psi) = -\frac{1}{|L|} \sum_{i \in L} \log \frac{\exp(z_{i, y_i})}{\sum_{c=1}^C \exp(z_{i, c})}$$

**Class-Balanced Cross-Entropy Loss**:
For imbalanced land-cover classes, class weight $w_c$ is calculated solely from the current labeled training pool $L$:
$$w_c = \frac{|L|}{C \cdot \max(1, N_c^{(L)})}$$
$$\mathcal{L}_{\text{CB}}(\psi) = -\frac{1}{|L|} \sum_{i \in L} w_{y_i} \log \frac{\exp(z_{i, y_i})}{\sum_{c=1}^C \exp(z_{i, c})}$$

Optimization uses AdamW ($\text{lr} = 5\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-4}$) with Cosine Annealing Learning Rate Schedule across 25 epochs per round.

---

### C. Validation Logit Calibration (Temperature Scaling)
To correct predictive overconfidence, a scalar temperature $T > 0$ is fitted on validation set $D_{\text{val}}$ logits by minimizing Negative Log-Likelihood:
$$\min_{T > 0} -\sum_{i \in D_{\text{val}}} \log \sigma \left( \frac{z(x_i)}{T} \right)_{y_i}$$
Calibrated class probabilities are computed as:
$$p_c(x_i) = \frac{\exp(z_i(c) / T)}{\sum_{j=1}^C \exp(z_i(j) / T)}$$

---

### D. Model B Query Engine (Calibrated Uncertainty + Labeled-Pool Diversity)

1. **Calibrated Least Confidence Uncertainty**:
   $$U(x_i) = 1 - \max_{c \in \{1 \dots C\}} p_c(x_i)$$

2. **Top-$M$ Candidate Filtering**:
   Unlabeled pool samples $U$ are ranked by $U(x_i)$. The top $M = \max(4k, 50)$ candidate indices $C_M \subset U$ are selected.

3. **Labeled-Pool-Aware Farthest-First Traversal**:
   Candidate feature embeddings $\bar{\phi}(x_i) = \frac{\phi(x_i)}{\|\phi(x_i)\|_2}$ are evaluated relative to existing labeled embeddings $L$:
   $$d(x_i, L) = \min_{l \in L} \| \bar{\phi}(x_i) - \bar{\phi}(l) \|_2$$
   The first candidate $x^{(1)} \in C_M$ selected is the one maximizing distance to the existing labeled pool:
   $$x^{(1)} = \arg\max_{x_i \in C_M} d(x_i, L)$$
   Subsequent candidates $x^{(m)}$ ($m=2 \dots k$) iteratively maximize distance to both labeled pool $L$ and previously selected candidates $\{x^{(1)} \dots x^{(m-1)}\}$.

---

### E. Cross-Sensor 1D Spectral Resampling & Feature Alignment
When transferring from Source Sensor (bands $B_{\text{src}}$) to Target Sensor (bands $B_{\text{tgt}}$):
1. **1D Deterministic Spectral Resampling**:
   Target spectrum $x_{\text{tgt}}(\lambda_1 \dots \lambda_{B_{\text{tgt}}})$ is linearly interpolated to match source wavelengths $\lambda_1 \dots \lambda_{B_{\text{src}}}$:
   $$x_{\text{resampled}}(\lambda_s) = \text{interp1d}(x_{\text{tgt}}(\lambda_t))$$
2. **Hidden Representation Transfer**:
   Target head $g_{\psi_{\text{tgt}}}$ initializes its first dense layer ($W_1, b_1$) directly from the trained source head $g_{\psi_{\text{src}}}$, enabling few-shot target active learning adaptation.
