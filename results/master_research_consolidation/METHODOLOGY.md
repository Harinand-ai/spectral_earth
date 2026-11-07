# ConformAL-HSI: Master Methodology Specification

**Document:** Formal Research Methodology & Experimental Protocol Specification  
**Date:** November 2026  
**Status:** Approved & Validated  

---

## 1. Mathematical Problem Formulation

Let $\mathcal{X} \subset \mathbb{R}^{B \times H \times W}$ denote the spatial-spectral hyperspectral input space with $B$ continuous spectral bands. A pretrained Spectral Foundation Model backbone $\Phi_\theta: \mathbb{R}^{B \times 15 \times 15} \to \mathbb{R}^D$ ($D=2048$) maps a localized spatial-spectral patch around pixel $i$ to a feature embedding $z_i = \Phi_\theta(x_i)$. The backbone weights $\theta$ are strictly frozen ($\nabla_\theta = 0$).

The goal of active learning is to iteratively construct an annotated training pool $\mathcal{L}_t \subset \mathcal{D}_{\text{train}}$ from an unlabeled candidate pool $\mathcal{U}_t = \mathcal{D}_{\text{train}} \setminus \mathcal{L}_t$ using minimal human labeling queries to train a lightweight neural classifier head $f_\psi: \mathbb{R}^D \to \Delta^C$ that minimizes risk on an independent test distribution:
$$\min_\psi \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{test}}} \left[ \ell(f_\psi(\Phi_\theta(x)), y) \right]$$

---

## 2. Dataset Partitions and Split Protocols

### 2.1 Standard Benchmark Partition (Cached Tensors)
Experiments are evaluated on pre-extracted feature tensors (`cache/preextracted_features/{slug}_{train,val,test}_feats.pt`):
- **Indian Pines (AVIRIS, 16 classes):** 7,174 Candidate Train (70.0%), 1,537 Val (15.0%), 1,538 Test (15.0%). Total: 10,249.
- **Salinas (AVIRIS, 16 classes):** 37,890 Candidate Train (70.0%), 8,119 Val (15.0%), 8,120 Test (15.0%). Total: 54,129.
- **Pavia University (ROSIS, 9 classes):** 29,943 Candidate Train (70.0%), 6,416 Val (15.0%), 6,417 Test (15.0%). Total: 42,776.

### 2.2 Autocorrelation Disclosure & Spatially Disjoint Holdout Protocol
- **Historical Random-Pixel Partition:** Shuffled ground-truth coordinates with $15 \times 15$ patches exhibit 100% spatial overlap between train and test patches (mean distance = 1.01 pixels).
- **Spatially Disjoint Block Partition:** A non-overlapping $8 \times 8$ spatial grid with a 7-pixel boundary guard band eliminates spatial patch overlap. On Pavia University, this cleanly isolates 15,846 training pixels from 2,133 test pixels with zero patch overlap while preserving 9/9 classes in both sets.

---

## 3. Classifier Head Architectures

1. **Standardized Head A (Primary Low-Budget Head):**
   $$\text{Input}(2048) \to \text{LayerNorm}(2048) \to \text{Linear}(2048 \to 256) \to \text{LayerNorm}(256) \to \text{GELU} \to \text{Linear}(256 \to C)$$
   - Trainable Parameters: $\approx 0.53\text{M}$.
   - Training Setup: AdamW ($\text{lr} = 5 \times 10^{-4}$, weight decay $10^{-4}$), Cosine Annealing, 15 epochs, batch size 64.
   - Properties: Deterministic, no running batch-stat dependencies, lowest Expected Calibration Error (4.4%–7.6%).

2. **Head B (High-Capacity Multi-Class Head):**
   $$\text{Input}(2048) \to L_2\text{-Norm} \to \text{Linear}(2048 \to 512) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.05) \to \text{Linear}(256 \to C)$$
   - Trainable Parameters: $\approx 1.19\text{M}$.
   - Properties: Higher non-linear representation capacity; superior AA and Macro-F1 on imbalanced datasets.

---

## 4. Split Conformal Prediction Engine

Split Conformal Prediction provides distribution-free finite-sample coverage guarantees using the held-out validation pool $\mathcal{D}_{\text{val}}$:
1. **Nonconformity Score:**
   $$s(x_i, y_i) = 1 - \hat{p}(y_i \mid x_i)$$
2. **Finite-Sample Quantile:**
   $$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^{n_{\text{cal}}}; \; \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right)$$
   where $\alpha = 0.10$ for $90\%$ nominal coverage.
3. **Prediction Set Construction:**
   $$\mathcal{C}(x) = \{c \in \{1, \dots, C\} : \hat{p}(y=c \mid x) \ge 1 - \hat{q}\}$$
4. **Conformal Active Query Scoring:**
   $$u_{\text{conf}}(x) = 0.6 \cdot \frac{|\mathcal{C}(x)| - 1}{\max(1, C - 1)} + 0.4 \cdot \left(1 - \text{margin}(\mathcal{C}(x))\right)$$

---

## 5. Active Learning Workflow

```
[Candidate Pool U]
        │
        ├──> Feature Extraction (Frozen 3D-ResNet50)
        │
        ├──> Classifier Inference: p(y | x)
        │
        ├──> Conformal Calibration on Val Pool -> q_hat
        │
        ├──> Acquisition Scoring (Margin / Conformal Cardinality)
        │
        ├──> Cosine Farthest-First Pool Diversity Filter
        │
        └──> Query Batch (k=20) -> Add to Labeled Pool L -> Retrain Head
```
