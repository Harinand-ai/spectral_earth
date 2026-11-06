# ConformAL-HSI: Classifier Head Architecture Ablation Study

**Framework:** ConformAL-HSI  
**Pretrained Backbone:** SpectralEarth 3D-ResNet50 ($L_2$-normalized 2048-dim embeddings, Frozen, $\nabla_\theta = 0$)  
**Benchmark Scope:** 4 Architectures $\times$ 4 Acquisition Strategies $\times$ 5 Label Budgets $\times$ 3 Datasets $\times$ Multi-Seed Evaluations  
**Hardware Environment:** NVIDIA GeForce RTX 2050 (4GB VRAM), PyTorch 2.x CUDA 12.x  

---

## 1. Architectural Configurations & Parameter Footprints

We systematically ablate the classifier head across four distinct architectural regimes, spanning from a minimalist linear baseline (0.53M parameters) to a deep multi-stage pyramid network (2.76M parameters):

| Identifier | Architecture Name | Structural Definition | Normalization | Activation & Regularization | Parameter Count ($C=16$) | Parameter Count ($C=9$) | Peak VRAM (MB) | Mean Latency / Run (s) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Head A** | Original Baseline 256 | $2048 \to 256 \to C$ | None | ReLU (no Dropout) | $528,656$ | $526,857$ | $37.9$ | $0.38\text{s}$ |
| **Head B** | Improved Head 512 | $2048 \to 512 \to C$ | BatchNorm1d ($512$) | GELU, Dropout ($p=0.10$) | $1,058,320$ | $1,054,729$ | $56.1$ | $0.48\text{s}$ |
| **Head C** | Wider Head 1024 | $2048 \to 1024 \to C$ | BatchNorm1d ($1024$) | GELU, Dropout ($p=0.20$) | $2,116,624$ | $2,109,449$ | $92.5$ | $0.62\text{s}$ |
| **Head D** | Deeper Pyramid 1024-512-256 | $2048 \to 1024 \to 512 \to 256 \to C$ | Two-Stage BN ($1024, 512$) | GELU, Dropout ($p_1=0.2, p_2=0.1$) | $2,761,488$ | $2,759,689$ | $114.6$ | $0.85\text{s}$ |

---

## 2. Empirical Performance Across Capacities

The table below contrasts the Overall Accuracy (OA, Mean $\pm$ Sample Std) across classifier heads at Budget $N=220$:

| Dataset | Acquisition Strategy | Head A (0.53M) | Head B (1.06M) | Head C (2.12M) | Head D (2.76M) | Capacity Gain (B vs A) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | Random Baseline | $38.60 \pm 1.98$ | $62.63 \pm 2.22$ | $63.78 \pm 2.01$ | $63.37 \pm 1.29$ | **+24.03%** |
| **Indian Pines** | Margin Sampling | $41.33 \pm 2.41$ | $66.86 \pm 1.25$ | $67.82 \pm 1.14$ | $65.81 \pm 1.48$ | **+25.53%** |
| **Indian Pines** | Margin + Cosine Div | $38.41 \pm 2.68$ | $65.34 \pm 1.70$ | $66.24 \pm 1.45$ | $64.12 \pm 1.62$ | **+26.93%** |
| **Indian Pines** | Adaptive Coverage AL | $36.79 \pm 2.15$ | $64.88 \pm 1.48$ | $65.71 \pm 1.52$ | $64.67 \pm 1.55$ | **+28.09%** |
| **Salinas** | Random Baseline | $41.47 \pm 2.30$ | $80.20 \pm 1.83$ | $80.64 \pm 1.62$ | $79.88 \pm 1.84$ | **+38.73%** |
| **Salinas** | Margin Sampling | $44.15 \pm 2.12$ | $81.72 \pm 0.94$ | $82.11 \pm 0.88$ | $80.95 \pm 1.15$ | **+37.57%** |
| **Salinas** | Margin + Cosine Div | $39.81 \pm 2.45$ | $81.18 \pm 1.12$ | $81.65 \pm 1.05$ | $80.52 \pm 1.20$ | **+41.37%** |
| **Salinas** | Adaptive Coverage AL | $38.92 \pm 2.28$ | $81.05 \pm 0.98$ | $81.42 \pm 0.92$ | $80.21 \pm 1.10$ | **+42.13%** |
| **Pavia Univ.** | Random Baseline | $60.40 \pm 1.15$ | $67.08 \pm 2.45$ | $67.48 \pm 2.20$ | $66.27 \pm 2.10$ | **+6.68%** |
| **Pavia Univ.** | Margin Sampling | $60.10 \pm 0.98$ | $69.36 \pm 1.02$ | $69.84 \pm 0.95$ | $68.44 \pm 1.05$ | **+9.26%** |
| **Pavia Univ.** | Margin + Cosine Div | $58.09 \pm 1.22$ | $68.42 \pm 1.18$ | $68.90 \pm 1.10$ | $67.45 \pm 1.25$ | **+10.33%** |
| **Pavia Univ.** | Adaptive Coverage AL | $57.34 \pm 1.40$ | $68.25 \pm 1.34$ | $68.80 \pm 1.28$ | $67.20 \pm 1.38$ | **+10.91%** |

---

## 3. Critical Analytical Insights

### 3.1 The Regularization Chasm (Head A vs. Head B)
- **Finding:** Head A collapses across all three datasets, failing to reach even $45\%$ OA on Indian Pines and Salinas, despite using the exact same frozen 2048-dimensional foundation model features.
- **Root Cause:** Head A lacks feature normalization (BatchNorm/LayerNorm) and stochastic regularization (Dropout). On small training sets ($N \in [100, 260]$), unnormalized linear layers overfit the dominant high-magnitude feature dimensions, leading to poor generalization.
- **Scientific Implication:** High-capacity foundation model representations **require properly regularized projection heads** to unlock their discriminative power in few-shot and active learning regimes.

### 3.2 Saturation and Diminishing Returns (Head B vs. Head C vs. Head D)
- **Head B (1.06M) vs. Head C (2.12M):** Doubling the width from 512 to 1024 yields marginal improvements ($+0.5\%$ to $+1.0\%$ OA), but requires doubling the dropout rate ($0.10 \to 0.20$) to prevent overfitting on $N=100$.
- **Head C (2.12M) vs. Head D (2.76M):** Adding depth (3 linear layers with 2 batch norm stages) actually causes a mild degradation ($-1.0\%$ to $-1.5\%$ OA) in low-budget regimes ($N=100-220$). In extreme few-shot scenarios, excess depth increases sample complexity and variance without providing additional representation capacity over the already-frozen 3D-ResNet50 backbone.
- **Optimal Architecture Recommendation:** **Head B (Improved 512)** represents the optimal Pareto trade-off between expressive capacity (1.06M parameters), regularization stability, memory efficiency ($56$ MB VRAM), and training speed ($0.48\text{s}$ per checkpoint).
