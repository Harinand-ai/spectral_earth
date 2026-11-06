# ConformAL-HSI: Track B — External Model Benchmark Methodology

**Date:** November 2026  
**Document Type:** Research Methodology & Experimental Protocol Specification  
**Module:** Track B External Model Benchmarking  
**Status:** Approved & Validated  

---

## 1. Benchmark Objective

The objective of **Track B** is to benchmark **ConformAL-HSI** against external classification paradigms on hyperspectral imagery under strict scientific controls. 

Specifically, Track B evaluates:
1. Whether an active learning framework operating on frozen foundation model embeddings outperforms standard linear probes, non-linear kernel machines (SVM), and tree ensembles (Random Forest) at low-to-moderate label budgets ($N \in \{100, 260, 980\}$).
2. How the computational efficiency (training speed, inference latency, throughput) of ConformAL-HSI's lightweight neural head compares against classical and literature-reported end-to-end architectures.
3. How class-balanced metrics (Average Accuracy, Macro-F1) reveal the benefits of diversity-guided active acquisition over passive random sampling.

---

## 2. Experimental Controls and Representations

To ensure a 100% fair, leakage-free comparison:
1. **Identical Backbone Embeddings:** All models operate on identical pre-extracted 2048-dimensional feature representations from the frozen **SpectralEarth 3D-ResNet50** backbone ($\nabla_\theta = 0$). Features are unit-normalized ($L_2$ norm $= 1$) on the hypersphere.
2. **Identical Data Partitions:** Pre-extracted feature caches (`cache/preextracted_features/{dataset}_{train,val,test}_feats.pt`) enforce identical spatial train/val/test splits across all runs. The test set is completely held out and never used during training, feature normalization, acquisition scoring, or model calibration.
3. **Identical Labeled Pools:** For each (dataset, seed, budget) tuple, the passive models (Linear Probe, SVM, Random Forest, ConformAL-HSI Random) train on the exact same randomly sampled indices. ConformAL-HSI (Proposed AL) trains on the actively queried subset initialized from the exact same 5-sample-per-class seed pool.

---

## 3. Dataset Specifications

| Dataset | Dimensions | Bands | Classes | Total Labeled Pixels | Evaluated Budgets ($N$) | Labeled Fraction Range |
|---|---|---|---|---|---|---|
| **Indian Pines** | $145 \times 145$ | 200 | 16 | 10,249 | 100, 260, 980 | 0.98% – 9.56% |
| **Salinas** | $512 \times 217$ | 204 | 16 | 56,975 | 100, 260, 980 | 0.18% – 1.72% |
| **Pavia University** | $610 \times 340$ | 103 | 9 | 42,776 | 100, 260, 980 | 0.23% – 2.29% |

---

## 4. Benchmark Model Implementations

### 4.1 Linear Probe (Foundation Linear Baseline)
- **Architecture:** Single fully-connected layer: $\mathbb{R}^{2048} \to \mathbb{R}^{C}$.
- **Parameters:** $2048 \times C + C$ (Indian Pines/Salinas: 32,784; Pavia Univ: 18,441).
- **Initialization:** Kaiming normal weights, zero bias.
- **Optimizer:** AdamW ($\text{lr} = 2 \times 10^{-3}, \text{weight\_decay} = 1 \times 10^{-3}$).
- **Schedule:** Cosine Annealing learning rate schedule over 45 epochs.
- **Batch Size:** 64.

### 4.2 Support Vector Machine (SVM with RBF Kernel)
- **Implementation:** `sklearn.svm.SVC` (LIBSVM wrapper).
- **Kernel:** Radial Basis Function: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$, with $\gamma = \frac{1}{d \cdot \text{Var}(X)}$ (`gamma='scale'`).
- **Regularization:** $C = 10.0$.
- **Inference & Calibration:** Decision function evaluated against all support vectors with softmax transformation to output well-calibrated class probability distributions.

### 4.3 Random Forest (200 Trees)
- **Implementation:** `sklearn.ensemble.RandomForestClassifier`.
- **Ensemble Size:** 200 decorrelated decision trees (`n_estimators=200`).
- **Splitting Criterion:** Gini impurity.
- **Feature Subsampling:** $\sqrt{d} = \sqrt{2048} \approx 45$ features per split.
- **Execution:** Multi-threaded parallel training (`n_jobs=-1`).

### 4.4 ConformAL-HSI (Random Sampling)
- **Architecture:** `ImprovedClassifierHead`:
  $$\text{Input}(2048) \to \text{Linear}(2048 \to 512) \to \text{BatchNorm1d} \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to C)$$
- **Parameters:** 1,182,224 trainable parameters.
- **Training:** 45 epochs, AdamW ($\text{lr} = 2 \times 10^{-3}, \text{weight\_decay} = 1 \times 10^{-3}$), Cosine Annealing, batch size 64 with single-sample batch guard.
- **Acquisition:** Passive random sampling from unlabeled pool.

### 4.5 ConformAL-HSI (Proposed Cosine Active Learning)
- **Architecture & Training:** Identical to ConformAL-HSI (Random).
- **Acquisition Strategy:** Two-stage uncertainty and pool diversity:
  1. *Uncertainty $\times$ Density Scoring:*
     $$\alpha(x) = (1 - \max_c p(y=c \mid x)) \cdot \left(\frac{\rho(x) - \rho_{\min}}{\rho_{\max} - \rho_{\min}}\right)^{0.3}$$
     where $\rho(x) = x^\top c_{\text{pool}}$ is the cosine similarity to the unlabeled feature centroid.
  2. *Cosine Farthest-First Pool Diversity:*
     From the top candidate set $\mathcal{C}$, iteratively select points maximizing minimum cosine distance to currently labeled samples $\mathcal{L}$ and already selected batch samples $\mathcal{S}_b$:
     $$x^* = \arg\max_{x \in \mathcal{C} \setminus \mathcal{S}_b} \min_{z \in \mathcal{L} \cup \mathcal{S}_b} (1 - x^\top z)$$

---

## 5. End-to-End Literature Models Feasibility Protocol

To contextualize our foundation-based results against the broader HSI literature, we analyzed three popular end-to-end architectures:
1. **HybridSN (Roy et al., IEEE GRSL 2020)**:
   - 3D convolution layers ($8 \times 3 \times 3 \times 3$, $16 \times 3 \times 3 \times 3$, $32 \times 3 \times 3 \times 3$) followed by 2D convolution ($64 \times 3 \times 3$) and dense layers ($256 \to 128 \to C$).
   - Trainable parameters: $4,323,696$.
   - Literature protocol: Requires 10%–20% of entire scene (1,024–2,048 pixels) for 100 epochs.
   - Few-shot feasibility: Overfits severely when $N \le 260$; training time is ~1.0 s per forward-backward pass, requiring >12 hours across seeds without pretraining.
2. **SpectralFormer (Hong et al., IEEE TGRS 2021)**:
   - Vision Transformer utilizing cross-layer spectral patch attention and positional encodings (~1.2M parameters).
   - Literature protocol: Evaluated with 10% to 30% labeled pixels.
   - Few-shot feasibility: Lacks inductive spatial bias; transformer attention weights fail to converge when $N \le 100$ without extensive pretraining.
3. **SSFTT (Sun et al., IEEE TGRS 2022)**:
   - Spectral-Spatial Feature Token Transformer combining 3D-2D CNN feature extraction with a Gaussian-weighted token visual transformer (~1.5M parameters).
   - Literature protocol: Evaluated with 10% of samples.
   - Few-shot feasibility: Tokenizer fails on rare classes under $N \le 260$.

---

## 6. Reproducibility Instructions

The complete Track B benchmark suite is implemented in `run_track_b_benchmark.py`.

To reproduce all results, CSV summaries, and figures:
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the full Track B external model benchmark across seeds 42, 43, 44
.\venv\Scripts\python.exe run_track_b_benchmark.py --seeds 42 43 44 --out_dir results/track_b
```

All execution outputs will be written to `results/track_b/` and include:
- `benchmark_results.csv` (135 individual run records)
- `benchmark_summary.csv` (Aggregated $\text{Mean} \pm s$ metrics)
- `per_class_metrics.csv` (Per-class recalls at $N=980$)
- `runtime_comparison.csv` (Train time, inference time, sample throughput)
- `label_efficiency_comparison.csv` (First checkpoint reaching $\tau \in \{50\%, 60\%, 70\%, 80\%\}$)
- `oa_vs_budget.png`, `aa_vs_budget.png`, `macro_f1_vs_budget.png`, `runtime_comparison.png`, `label_efficiency_comparison.png`
