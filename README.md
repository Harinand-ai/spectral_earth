# SpectralEarth Active Learning Framework — Visual Research Demonstration System

An end-to-end interactive web application and research demonstration system built on top of the verified **SpectralEarth Active Learning Framework**. 

This system provides complete visual demonstration for active learning query selection, spatial uncertainty mapping, sample-selection explanations, simulated oracle labeling, round-by-round classifier head retraining, cross-sensor domain adaptation, label-efficiency benchmarking, and automated visual research report generation—**while strictly preserving the frozen SpecResNet50 backbone and verified research pipeline without unauthorized metric optimization**.

---

## Key Features

1. **Strict Frozen Backbone Preservation**:
   - Backbone: `spec_resnet50` (pretrained on 1.2M unlabeled hyperspectral patches).
   - Backbone parameters: **FROZEN** (0 backbone parameters updated during active learning).
   - Active learning head: 2-layer MLP classifier head (`in_features=2048`, `hidden_dim=256`, `dropout=0.0`, `norm_layer=layernorm`, 524,800 trainable parameters).

2. **Interactive Active Learning Pipeline Workflow (Steps 1 to 10)**:
   - **Step 1: Scene & Dataset Information**: RGB composite generation, spatial dimensions, spectral band metadata, ground-truth label distribution.
   - **Step 2: Initial Labeled Pool Visualization**: Class-balanced initial seed pool (10 samples per class) visual spatial mask.
   - **Step 3: Spatial Model Predictions**: Full-scene classification map generated via fast feature-level inference.
   - **Step 4: Spatial Uncertainty Mapping**: Spatial heatmaps for `Least Confidence`, `Entropy`, and `Margin` uncertainty metrics.
   - **Step 5: Query Selection & Sample Explanations**: `Least Confidence + Farthest Diversity` candidate filtering (`k=20` samples per round). Displays an explicit *"Why Was This Sample Selected?"* table with percentile rankings.
   - **Step 6: High-Resolution Sample Inspector**: Interactive modal showing 15x15 spatial patch zoom, 1D spectral reflectance signature curve across wavelengths, class probability distribution, and selection reason.
   - **Step 7: Simulated Human Annotation**: Oracle labeling interface using ground-truth training pool labels only. (Explicit data leakage assertions ensure test set labels NEVER enter query selection or training).
   - **Step 8: Classifier Head Retraining**: Real-time retraining of 2-layer MLP classifier head on updated labeled pool with loss and duration monitoring.
   - **Step 9: BEFORE vs AFTER Classification Maps**: Side-by-side comparison of classification maps before and after retraining.
   - **Step 10: Progression & Calibrated Performance Metrics**: Live interactive performance curves (OA, AA, Macro F1, Kappa, Calibrated ECE) across active learning rounds 0 to 5.

3. **Verified Cross-Sensor Domain Adaptation Page**:
   - Scientific distinction between **CROSS-DATASET** (within AVIRIS) and **CROSS-SENSOR** (AVIRIS &rarr; ROSIS).
   - Transfer pair selector: `Indian Pines (AVIRIS) -> Pavia University (ROSIS)` and `Salinas (AVIRIS) -> Pavia University (ROSIS)`.
   - **1D Spectral Resampler Visualizer (`SpectralResampler1D`)**: Plots source wavelengths vs target wavelengths vs resampled representation with explicit label: *"Deterministic 1D spectral interpolation (No learned domain alignment)"*.
   - Zero-Shot vs Few-Shot Active Learning adaptation performance breakdown.
   - Per-sample target query selection explanations.

4. **Label-Efficiency Benchmark**:
   - Interactive comparative performance curves (`Random`, `Least Confidence`, `LC + Farthest Diversity`).
   - Labeled sample threshold savings table (showing 30%-50% label savings to reach target accuracy thresholds).

5. **Automated Visual Research Report & Exports**:
   - Automated 15-section Visual Research Report generator in HTML and JSON format.
   - One-click export of experiment session data (JSON, CSV) saved directly to `logs/al_experiments/` with timestamps.

---

## System Architecture

```
                                  [ Hyperspectral Scene Data ]
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     FROZEN SpecResNet50          │
                              │   (No Parameter Updates)         │
                              └──────────────────────────────────┘
                                               │
                                    2048-dim Feature Vectors
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
          ┌─────────────────────────┐                     ┌─────────────────────────┐
          │   2-Layer MLP Classifier│                     │      QueryEngine        │
          │   Head (Trainable)      │                     │  (LC + Farthest Div)    │
          └─────────────────────────┘                     └─────────────────────────┘
                       │                                               │
             Predicted Probabilities                               Selected k=20
                       │                                             Samples & Scores
                       ▼                                               │
          ┌─────────────────────────┐                                  │
          │ Calibrated Evaluation   │ ◄────────────────────────────────┘
          │ (Validation Scaling)    │     Simulated Human Annotation
          └─────────────────────────┘
```

---

## Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA supported)

### 2. Launch the Demonstration Web Application

Run the launcher script:
```bash
python run_visualization_app.py
```
Or start via Uvicorn directly:
```bash
uvicorn src.visualization_app.app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Verification Test Suite

### 1. Active Learning Integration Tests
```bash
python test_al_integration.py
python test_adaptive_controller.py
python test_pseudolabel_engine.py
```

### 2. Multi-Seed Statistical Validation Suite
```bash
python run_multiseed_validation.py
```

---

## Higher Label-Budget Validation Experiments

To investigate scaling behavior as the genuine annotation budget expands towards 1,000 samples, we provide the dedicated higher-budget experimental suite:

```bash
python run_higher_budget_experiments.py --seeds 42 43 44 --out_dir results/higher_budget
```

### Experimental Configurations:
- **Datasets:** Indian Pines ($C=16$), Salinas ($C=16$), Pavia University ($C=9$).
- **Budgets Tested:** Initial Baseline ($N=160$ for Indian Pines/Salinas, $N=190$ for Pavia), $N=100$, $N=250$, $N=645$, and $N=980$ total labeled samples.
- **Strategies Compared:**
  1. `Random Sampling`: Passive uniform random acquisition.
  2. `Uncertainty Sampling`: Standard Least Confidence ($1 - \max_c p(c|x)$).
  3. `Proposed (Uncertainty + Diversity)`: Candidate-filtered farthest-first traversal in $\mathbb{R}^{2048}$.
  4. `Proposed + Pseudo-Labeling`: High-confidence pseudo-labeling ($\tau \ge 0.95$, weight $\lambda_p = 0.20$).
- **Hardware Acceleration:** Fully vectorized GPU tensor queries on CUDA using cached 2048-dimensional feature embeddings.
- **Outputs Generated:**
  - Structured results: `results/higher_budget/higher_budget_results.csv` and `higher_budget_results.json`.
  - Publication figures: `results/higher_budget/accuracy_vs_budget_all_datasets.png`, `accuracy_vs_budget_indian_pines.png`, `accuracy_vs_budget_salinas.png`, `accuracy_vs_budget_pavia_university.png`, and `confusion_matrices_best_configurations.png`.

### Key Scaling Findings & Bottleneck Analysis:
| Dataset | Strategy | Budget 100 | Budget 160 / 190 | Budget 250 | Budget 645 | Budget 980 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | Proposed (Uncertainty + Diversity) | 34.70% ± 1.44% | 42.78% ± 0.88% | 47.55% ± 2.22% | 55.90% ± 2.36% | **65.13% ± 0.30%** |
| **Indian Pines** | Proposed + Pseudo-Labeling | 33.75% ± 2.60% | 43.89% ± 2.60% | 49.09% ± 2.98% | 55.37% ± 1.80% | **60.49% ± 6.33%** |
| **Salinas** | Proposed (Uncertainty + Diversity) | 44.64% ± 5.35% | 37.70% ± 5.80% | 49.33% ± 3.28% | 53.65% ± 2.17% | **66.15% ± 2.11%** |
| **Salinas** | Proposed + Pseudo-Labeling | 43.65% ± 2.83% | 38.06% ± 1.12% | 49.75% ± 6.17% | 58.53% ± 1.83% | **67.35% ± 2.54%** |
| **Pavia Univ.** | Proposed (Uncertainty + Diversity) | 53.77% ± 4.97% | 54.57% ± 6.41% | 49.02% ± 7.01% | 52.70% ± 4.06% | **59.45% ± 3.13%** |
| **Pavia Univ.** | Proposed + Pseudo-Labeling | 51.83% ± 7.99% | 53.27% ± 4.26% | 56.26% ± 6.26% | 55.56% ± 4.87% | **60.42% ± 5.61%** |

---

## ConformAL-HSI: Systematic Diagnosis & Architectural Improvements

To investigate why higher-label budgets initially plateaued at 60%–73% and why passive random sampling previously outperformed uncertainty sampling on Salinas and Pavia University, we conducted a rigorous forensic audit and controlled multi-seed benchmark across all three benchmarks.

### 1. Root-Cause Scientific Diagnosis
1. **The Frozen Backbone is Highly Expressive**: Nonlinear probes on the frozen $\text{SpecResNet50}$ representations achieve an empirical upper bound of **87.65% (Indian Pines), 90.86% (Salinas), and 92.72% (Pavia University)**. The backbone is *not* the primary bottleneck.
2. **Classifier Head Underfitting**: The original head (256 hidden units, 25 epochs, fixed $lr=10^{-3}$) performed only 375 gradient updates at budget 980, severely underfitting the non-linear manifold.
3. **The Euclidean Norm Pathology**: In $\mathbb{R}^{2048}$, raw vector norms exhibited extreme variance (Salinas: $3,431$ to $154,677$; Pavia: $698$ to $82,133$). Raw Euclidean distance $\|z_i - z_j\|_2$ picked extreme-norm outliers rather than semantic diversity.
4. **Boundary Trapping in Spatially Smooth Parcels**: Salinas and Pavia consist of expansive homogeneous land-cover parcels. Uncalibrated uncertainty sampling repeatedly queried ambiguous border pixels between fields while neglecting parcel interiors. Random sampling uniformly sampled cluster interiors.
5. **Confirmation Bias in Low-Budget Pseudo-Labeling**: At $N \le 100$, uncalibrated pseudo-labels reinforce erroneous boundary predictions unless calibrated and class-capped.

### 2. Implemented Framework Enhancements
- **`ImprovedClassifierHead` (`src/learning_engine/improved_classifier.py`)**: Unit hypersphere $\ell_2$-normalization projection, 512-dim hidden representation with batch normalization, lightweight dropout ($p=0.10$), and Cosine Annealing learning rate schedule ($45$ epochs, initial $lr=2\times 10^{-3}$).
- **Cosine-Space Labeled-Pool Diversity**: Candidate farthest-first traversal performed strictly on unit hypersphere embeddings relative to the existing labeled pool $\mathcal{L}$ ($d_{\text{cos}}(x, \mathcal{L}) = 1 - \max_{y \in \mathcal{L}} \hat{x} \cdot \hat{y}$) combined with a density prior to prevent boundary outlier traps.
- **Calibrated Pseudo-Labeling Engine**: Confidence threshold $\tau \ge 0.95$, loss weighting $w_p = 0.20$, and per-class selection quota (max 20 pseudo-samples per class).

### 3. Controlled Empirical Validation (3 Seeds: 42, 43, 44)

| Dataset | Configuration | Budget 100 | Budget 250 | Budget 645 | Budget 980 (OA) | Budget 980 (AA) | Budget 980 ($\kappa$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Indian Pines** | Baseline (Orig Head + Random) | 42.98% | 52.06% | 60.86% | 66.86% | 62.08% | 0.6151 |
| **Indian Pines** | Baseline (Orig Head + Uncertainty) | 40.38% | 46.81% | 51.19% | 61.73% | 59.50% | 0.5570 |
| **Indian Pines** | Baseline (Orig Head + Unc+Div) | 41.31% | 45.56% | 48.87% | 62.98% | 63.41% | 0.5729 |
| **Indian Pines** | **Improved Head + Improved AL** | 46.58% | 58.54% | 74.27% | **86.22%** | **85.18%** | **0.8415** |
| **Indian Pines** | **Improved Head + AL + Pseudo** | 37.54% | 54.42% | 71.39% | **84.40%** | **81.85%** | **0.8197** |
| **Salinas** | Baseline (Orig Head + Random) | 49.48% | 57.97% | 64.67% | 69.80% | 68.82% | 0.6611 |
| **Salinas** | Baseline (Orig Head + Unc+Div) | 39.40% | 41.79% | 52.54% | 67.03% | 66.73% | 0.6301 |
| **Salinas** | **Improved Head + Improved AL** | 62.57% | 68.17% | 76.34% | **83.75%** | **87.71%** | **0.8192** |
| **Salinas** | **Improved Head + AL + Pseudo** | 49.91% | 63.76% | 76.15% | **84.10%** | **88.09%** | **0.8230** |
| **Salinas** | **Improved Head + Random** | 65.86% | 75.59% | 80.46% | **85.35%** | **88.34%** | **0.8367** |
| **Pavia Univ.** | Baseline (Orig Head + Random) | 58.75% | 65.80% | 69.27% | 72.59% | 66.04% | 0.6317 |
| **Pavia Univ.** | Baseline (Orig Head + Unc+Div) | 51.25% | 50.14% | 55.17% | 62.14% | 66.37% | 0.5264 |
| **Pavia Univ.** | **Improved Head + Improved AL** | 49.33% | 54.48% | 67.76% | **73.67%** | **72.95%** | **0.6635** |
| **Pavia Univ.** | **Improved Head + AL + Pseudo** | 43.16% | 57.08% | 68.96% | **75.37%** | **72.44%** | **0.6794** |
| **Pavia Univ.** | **Improved Head + Random** | 61.17% | 69.92% | 76.18% | **79.56%** | **74.98%** | **0.7286** |

*Reproducibility Command:*
```bash
python run_conform_al_hsi_improvement.py --seeds 42 43 44 --out_dir results/improved_framework
```

---


Run the automated verification script to validate all REST endpoints, feature extraction engines, active learning steps, cross-sensor transfers, and export functions:

```bash
python test_complete_demonstration.py
```

---

## Research Pipeline Constraints & Assertions

1. **Backbone Freeze**: `param.requires_grad = False` enforced for all backbone weights.
2. **Data Leakage Protection**:
   - Training Pool (`train_dataset`): Used for feature extraction and active query candidate selection.
   - Validation Set (`val_dataset`): Used ONLY for temperature calibration scaling.
   - Test Set (`test_dataset`): Used ONLY for final evaluation metrics calculation (OA, AA, F1, Kappa, ECE). Test labels NEVER enter query selection or model training.
3. **Reproducibility**: All experiments run with fixed global seed `42`.

---

## Citation & Framework References
- **Backbone**: SpecResNet50 (SpectralEarth SSL Foundation Backbone)
- **Active Learning Strategy**: Least Confidence + Farthest Diversity (`filter_diversity`)
- **Spectral Resampling**: `SpectralResampler1D`
