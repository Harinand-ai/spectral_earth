# ConformAL-HSI: Low-Label Efficiency Optimization Methodology & Experimental Protocol

**Date:** November 2026  
**Document Type:** Research Protocol & Methodological Specification  
**Project:** ConformAL-HSI (Hyperspectral Active Learning on Frozen SpectralEarth)  
**Research Priority:** Low-Label Efficiency ($N \in \{100, 140, 180, 220, 260\}$)  
**Status:** Approved & Actively Executing  

---

## 1. Primary Research Objective

The central objective of this research phase is **LABEL EFFICIENCY AT LOW BUDGETS**:
$$\text{Achieve competitive hyperspectral classification accuracy using the absolute minimum number of genuinely labeled training samples.}$$

Rather than prioritizing high-budget saturation ($N \ge 645$), we focus systematically on the **ultra-low to moderate annotation regime**:
$$\mathcal{B}_{\text{low}} = \{100, 140, 180, 220, 260\} \text{ labeled samples}$$

For Indian Pines ($10,249$ pixels) and Salinas ($56,975$ pixels), starting from $C \times 5 = 80$ initial seed labels, this corresponds to acquiring only **$20, 60, 100, 140, 180$ active queries** ($0.18\%–2.54\%$ of the scene). For Pavia University ($42,776$ pixels, 9 classes), starting from $45$ initial seed labels, this corresponds to acquiring **$55, 95, 135, 175, 215$ active queries** ($0.23\%–0.61\%$ of the scene).

---

## 2. Invariant Architectural Controls

To ensure strict scientific validity and prevent confounding factors:
1. **Frozen Foundation Backbone**: The pretrained **SpectralEarth 3D-ResNet50** feature extractor remains 100% frozen ($\nabla_\theta = 0$). No backbone fine-tuning, adapter retraining, or weights modification occurs.
2. **Fixed Feature Manifold**: All strategies operate on identical pre-extracted 2048-dimensional feature representations, normalized onto the unit hypersphere ($L_2$ norm $= 1$).
3. **Fixed Classifier Architecture**: All acquisition comparisons use an identical `ImprovedClassifierHead`:
   $$\text{Input}(2048) \to \text{Linear}(2048 \to 512) \to \text{BatchNorm1d} \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to C)$$
4. **Fixed Training Procedure**: At each budget checkpoint, the classifier is trained for **45 epochs** using AdamW ($\text{lr} = 2 \times 10^{-3}, \text{weight\_decay} = 1 \times 10^{-3}$) with Cosine Annealing learning rate scheduling and mini-batch size 64.
5. **Intermediate Model Guard**: During active acquisition loops, intermediate models trained to score candidate points use **20 epochs** to provide rapid, computationally lightweight uncertainty estimates.
6. **Strictly Held-Out Test Set**: The test set is completely isolated and never accessed during acquisition scoring, feature centering, temperature calibration, or hyperparameter selection.

---

## 3. Acquisition Strategies Taxonomy

### A. Uncertainty-Based Acquisition
1. **Least Confidence Sampling**:
   $$u_{\text{LC}}(x) = 1.0 - \max_{c} p(c \mid x)$$
   Selects samples where the model's most confident prediction is lowest.
2. **Margin Sampling**:
   $$u_{\text{Margin}}(x) = 1.0 - (p_{(1)}(x) - p_{(2)}(x))$$
   Focuses on points closest to decision boundaries between the top two competing classes.
3. **Entropy Sampling**:
   $$u_{\text{Entropy}}(x) = -\frac{1}{\log C} \sum_{c=1}^C p(c \mid x) \log p(c \mid x)$$
   Measures overall dispersion across the full class probability vector.
4. **BALD (Bayesian Active Learning by Disagreement)**:
   Measures epistemic uncertainty (model parameter uncertainty) by estimating mutual information between predictions and model parameters via Monte Carlo Dropout ($T=10$ stochastic forward passes):
   $$\text{BALD}(x) = H\left[\frac{1}{T}\sum_{t=1}^T p^{(t)}(y \mid x)\right] - \frac{1}{T}\sum_{t=1}^T H\left[p^{(t)}(y \mid x)\right]$$
   Isolates model ambiguity from irreducible data noise.

---

### B. Diversity-Based Acquisition
1. **Cosine Diversity Farthest-First**:
   Greedily selects candidates that maximize the minimum cosine distance to currently labeled samples $\mathcal{L}$ and already-selected batch samples $\mathcal{S}_b$:
   $$x^* = \arg\max_{x \in \mathcal{C} \setminus \mathcal{S}_b} \min_{z \in \mathcal{L} \cup \mathcal{S}_b} (1 - x^\top z)$$
2. **K-Center Greedy (Core-Set Facility Location)**:
   Constructs a core-set minimizing the maximum distance from any unlabeled point to the labeled set on the unit hypersphere:
   $$\min_{\mathcal{S}} \max_{x \in \mathcal{U}} \min_{z \in \mathcal{L} \cup \mathcal{S}} \|f(x) - f(z)\|_2$$
3. **Cluster Diversity (Spherical K-Means)**:
   Partitions candidate representations into $k$ clusters on the unit hypersphere and selects the densest representative sample per cluster.

---

### C. Gradient-Based Acquisition (BADGE)
**BADGE (Batch Active learning by Diverse Gradient Embeddings)**:
Combines uncertainty and diversity by computing the loss gradient with respect to the final linear layer weights:
$$g(x) = (p(x) - e_{\hat{y}(x)}) \otimes f(x)$$
- The norm $\|g(x)\|_2 \approx \sqrt{2}(1 - p_{\hat{y}})$ directly captures **prediction uncertainty**.
- The directional alignment $g(x_1)^\top g(x_2)$ captures **feature and class diversity**.
- **Memory Optimization for 4GB RTX 2050 GPU**:
  To prevent out-of-memory errors on large pools ($50,000 \times 16 \times 2048 \times 4\text{ B} = 6.5\text{ GB}$), we pre-filter candidates by margin uncertainty to a candidate set $M = \max(4k, 200)$, computing $g(x)$ exclusively on $M$ ($26\text{ MB}$), followed by greedy farthest-first traversal.

---

### D. Adaptive Class-Coverage Hybrid Acquisition
To address **dominant-class starvation** (observed on Pavia Meadows) and **minority-class under-sampling** (observed on Indian Pines), we formulate an adaptive multi-objective acquisition score:
$$S(x) = \alpha U(x) + \beta D(x) + \gamma C(x)$$
Where:
- $U(x)$: Normalized uncertainty score (Margin or Entropy).
- $D(x)$: Normalized cosine distance to currently labeled pool.
- $C(x)$: **Class Coverage Factor**, derived strictly from training-pool information:
  $$C(x) = \frac{1}{\sqrt{N_{\hat{y}(x)} + 1}}$$
  where $N_{\hat{y}(x)}$ is the number of currently labeled samples belonging to the predicted class $\hat{y}(x) = \arg\max_c p(c \mid x)$.
  - If a predicted class is severely under-represented in the current training set, $C(x)$ boosts its priority.
  - If a class already has abundant labels, $C(x)$ penalizes further redundant queries.

---

## 4. Evaluation Metrics & Label Efficiency Determination

### Primary Evaluation: Measured Checkpoint Label Savings
Label savings are calculated exclusively at **actual measured experimental checkpoints** ($\mathcal{B}_{\text{low}} = \{100, 140, 180, 220, 260\}$):
$$\text{Label Savings (\%)} = \left(\frac{N_{\text{random}} - N_{\text{method}}}{N_{\text{random}}}\right) \times 100$$
- If Method A reaches target accuracy $\tau$ at $N=140$ while Random requires $N=220$:
  $$\text{Savings} = \frac{220 - 140}{220} \times 100 = \mathbf{+36.36\%} \quad (80\text{ fewer labels})$$
- If Random reaches $\tau$ at $N=180$ while Method A requires $N=220$:
  $$\text{Savings} = \frac{180 - 220}{180} \times 100 = \mathbf{-22.22\%} \quad (40\text{ extra labels})$$

### Complete Metric Suite
1. **Overall Accuracy (OA)**: Area-weighted pixel classification accuracy.
2. **Average Accuracy (AA)**: Class-balanced unweighted recall across all classes.
3. **Macro-F1 Score**: Harmonic mean of class-wise precision and recall.
4. **Cohen's Kappa ($\kappa$)**: Chance-corrected agreement statistic.
5. **Expected Calibration Error (ECE)**: Probability reliability assessment.
6. **Acquisition Latency**: Wall-clock duration of query selection step.

---

## 5. Phased Research Execution Plan

```
[Phase 1: Baselines Verification]
  - Random Sampling
  - Proposed Cosine AL
  - Least Confidence
  - Margin Sampling
  - Entropy Sampling
         │
         ▼
[Phase 2: Advanced Acquisition Methods]
  - K-Center Greedy (Core-Set)
  - BALD (MC Dropout)
  - BADGE (Gradient Embeddings)
  - Cluster Diversity
         │
         ▼
[Phase 3: Hybrid & Adaptive Optimization]
  - Margin + Cosine Diversity
  - BALD + Cosine Diversity
  - Adaptive Class-Coverage AL (S(x) = αU + βD + γC)
         │
         ▼
[Phase 4: Calibration & Pseudo-Labeling Analysis]
  - Validation Temperature Scaling (ECE & Brier Score)
  - Conservative Class-Specific Pseudo-Labeling
         │
         ▼
[Phase 5: Multi-Seed Statistical Confirmation]
  - Top 3 Candidate Strategies vs. Random across 5 Seeds [42, 43, 44, 45, 46]
```

---

## 6. Computational Feasibility & Technical Controls

All experiments run locally under the following system specifications:
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4GB VRAM).
- **Environment**: Python 3.11/PyTorch 2.x with CUDA 12.x.
- **Memory Safety**: Mini-batch vectorization (batch size 2048 for inference, batch size 64 for classifier training).
- **Zero Backbone Overhead**: All 2048-dimensional embeddings are pre-extracted and cached in `cache/preextracted_features/`, eliminating repeated backbone forward passes.

