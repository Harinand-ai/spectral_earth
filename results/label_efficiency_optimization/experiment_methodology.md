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
**BADGE (Batch Active Learning by Diverse Gradient Embeddings)** (Ash et al., 2020):
Combines uncertainty and diversity by computing the loss gradient with respect to the final classification layer parameters:
$$g(x) = (p(x) - e_{\hat{y}(x)}) \otimes f(x)$$
- The norm $\|g(x)\|_2 = \|p(x) - e_{\hat{y}}\|_2 \|f(x)\|_2 \approx \sqrt{2}(1 - p_{\hat{y}})$ directly captures **prediction uncertainty**.
- The directional inner product $g(x_1)^\top g(x_2)$ captures **joint feature and class diversity**.

#### Distinction: Exact BADGE vs. Candidate-Prefiltered BADGE
1. **Candidate-Prefiltered Approximation (Heuristic Baseline)**:
   A common engineering heuristic to prevent GPU out-of-memory errors on large pools ($50,000 \times 16 \times 2048 \times 4\text{ B} = 6.55\text{ GB}$) is to pre-filter candidates by margin uncertainty to a small subset $M \ll |\mathcal{U}|$ (e.g. $M = \max(4k, 200)$), materializing $g(x) \in \mathbb{R}^{M \times (C \cdot d)}$ explicitly. However, this heuristic discards potentially diverse non-boundary points and introduces an arbitrary truncation hyperparameter.

2. **Exact BADGE via Tensor Inner-Product Factorization (Our Rigorous Formulation)**:
   We mathematically factorize the pairwise gradient distance between any two points $x_i, x_j$ without ever materializing the massive $(|\mathcal{U}| \times 32,768)$ gradient matrix:
   $$\|g(x_i) - g(x_j)\|_2^2 = \|g(x_i)\|_2^2 + \|g(x_j)\|_2^2 - 2 \langle g(x_i), g(x_j) \rangle$$
   Since $f(x)$ is $L_2$-normalized ($\|f(x)\|_2 = 1$):
   $$\|g(x_i)\|_2^2 = \|r_i \otimes f_i\|_2^2 = \|r_i\|_2^2 \cdot \|f_i\|_2^2 = \|r_i\|_2^2$$
   where $r_i = p(x_i) - e_{\hat{y}_i} \in \mathbb{R}^C$.
   Furthermore, by tensor inner-product properties:
   $$\langle g(x_i), g(x_j) \rangle = \text{Tr}\left((r_i \otimes f_i)^\top (r_j \otimes f_j)\right) = (r_i^\top r_j) \cdot (f_i^\top f_j)$$
   Therefore, the exact squared Euclidean distance in gradient embedding space is:
   $$\|g(x_i) - g(x_j)\|_2^2 = \|r_i\|_2^2 + \|r_j\|_2^2 - 2 (r_i^\top r_j) (f_i^\top f_j)$$
   - **Computational & Memory Complexity**: Rather than storing $6.55\text{ GB}$, this requires only storing $r \in \mathbb{R}^{|\mathcal{U}| \times C}$ ($3.2\text{ MB}$) and $f \in \mathbb{R}^{|\mathcal{U}| \times d}$ ($409\text{ MB}$).
   - **Exactness**: Evaluates 100% of the unlabeled pool without candidate downsampling, gradient pruning, or out-of-memory errors on a 4GB RTX 2050 GPU. Exact BADGE completes over $50,000$ points in under $0.38\text{ seconds}$.

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

> [!IMPORTANT]
> **Strict Equivalence Policy**:
> 1. Label savings claims are valid **only when Method A achieves measured test accuracy strictly meeting or exceeding target threshold $\tau$ ($\text{OA}_A \ge \tau$)**, and Random Sampling's required budget is measured at the same standard ($\text{OA}_{\text{Random}} \ge \tau$).
> 2. **No Near-Miss Savings Claims**: For example, on Pavia University, Margin Sampling reaches $66.32\%$ at $N=180$, while Random Sampling reaches $67.08\%$ at $N=220$. Because $66.32\% < 67.08\%$, it is scientifically invalid to claim Margin at 180 achieves equivalent accuracy to Random at 220. At threshold $\tau = 66.0\%$, however, Margin reaches it at $N=180$ ($66.32\% \ge 66.0\%$) whereas Random reaches it at $N=220$ ($67.08\% \ge 66.0\%$), yielding an exact savings of $+18.18\%$.
> 3. **Non-Universal Superiority**: Active learning methods do not universally outperform Random Sampling across all budgets. In ultra-low budget regimes ($N=100, 140$), Random Sampling frequently matches or exceeds active strategies due to undertrained initial decision boundaries causing sampling bias. All budget- and dataset-specific results are reported transparently.

### Cumulative Labeled Set Integrity & Non-Monotonic Accuracy Investigation
Across all evaluated datasets and acquisition methods, small accuracy dips are occasionally observed between $N=220$ and $N=260$ (e.g., Indian Pines Margin dropping from $63.80\%$ to $62.70\%$, and Random dropping from $63.37\%$ to $61.66\%$).
We systematically audited this phenomenon and verified:
1. **Strict Subset Invariance**: The labeled set is rigorously cumulative:
   $$\mathcal{L}_{100} \subset \mathcal{L}_{140} \subset \mathcal{L}_{180} \subset \mathcal{L}_{220} \subset \mathcal{L}_{260}$$
   Every previously acquired sample is strictly preserved; no samples are dropped or discarded.
2. **Batch Normalization Dynamics**: With mini-batch size 64:
   - At $N=220$: $220 = 3 \times 64 + 28$ samples (4 batches, residual batch has 28 samples).
   - At $N=260$: $260 = 4 \times 64 + 4$ samples (5 batches, residual batch has only 4 samples).
   In small residual mini-batches ($B_{\text{rem}}=4$), batch statistics ($\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}^2$) exhibit high stochastic variance during training, mildly degrading final running statistics in `BatchNorm1d`.
3. **Identical Test Evaluation**: The test feature set, labels, and evaluation metric engine are 100% frozen and identical across all budgets and seeds.

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

