# ADAPTIVE QUERY CONTROLLER EXPERIMENT REPORT

**Date**: 2026-08-23  
**Project**: SpectralEarth Active Learning Framework  
**Author/Auditor**: Building & Implementation Agent (Antigravity)  
**Status**: Official Experimental Evaluation Report  

---

## 1. Executive Summary & Final Classification

In accordance with the **SpectralEarth 3-Hour Final System Optimization & Demonstration Directive**, a state-driven, deterministic, and computationally lightweight controller (`AdaptiveQueryController`) was designed, implemented in `src/query_engine/query_engine/adaptive_controller.py`, integrated into `QueryEngine`, and benchmarked against standard baselines (`Random`, `Least Confidence`, and `LC + Farthest Diversity`).

### **FINAL EXPERIMENTAL CLASSIFICATION**:
> **`CONDITIONAL KEEP / REJECT FOR GENERAL METHODOLOGY, BUT KEEP IMPLEMENTED AS AN ABLATION OPTION`**

### **Key Rationale**:
1. **Salinas**: Fixed `LC + Farthest Diversity` achieves superior overall accuracy (**72.28% OA** vs 68.87% for Adaptive Controller) and higher Macro F1 (0.7573 vs 0.7281).
2. **Indian Pines**: Adaptive Controller achieves slightly higher Average Accuracy (**68.70% AA** vs 68.03%), but fixed `LC + Farthest Diversity` yields better Expected Calibration Error (**0.0562** vs 0.0692).
3. **Pavia University**: Fixed `LC + Farthest Diversity` outperforms the Adaptive Controller in Macro F1 (**0.6036** vs 0.5673) and Overall Accuracy (**58.89%** vs 55.26%).
4. **Scientific Lesson**: Empirical results confirm that blindly adjusting dynamic weights across rounds adds complexity without consistently outperforming the fixed candidate-filtered diversity (`filter_diversity`) strategy. Fixed `filter_diversity` remains the primary recommended research strategy.

---

## 2. Adaptive Query Controller Design & Formulation

The `AdaptiveQueryController` operates without test labels or ground-truth confusion matrices, strictly relying on non-test model signals:

### State Signals Evaluated:
1. **AL Progress ($r / R$)**: Current round ratio $r / (R-1) \in [0, 1]$.
2. **Mean Predictive Confidence ($\bar{C}$)**: Average maximum probability $\frac{1}{N_u} \sum_{i=1}^{N_u} \max_c P(y=c|x_i)$ across unlabeled pool.
3. **Representation Coverage Gap ($\max D_{\text{labeled}}$)**: Maximum distance of candidate embeddings to existing labeled representations.
4. **Hard-Region Ambiguity Score ($H(x)$)**: Product of least confidence uncertainty and labeled pool distance ($U(x) \times D_{\text{labeled}}(x)$).

### Weight Schedule & Equations:
$$\text{Progress } p = \frac{r}{R - 1}$$
$$w_U^{\text{base}} = 0.30 + p \cdot (0.70 - 0.30), \quad w_D^{\text{base}} = 1.0 - w_U^{\text{base}}, \quad w_H^{\text{base}} = 0.15$$

$$\text{If } \bar{C} < 0.45 \implies w_D \leftarrow w_D + 0.10 \times \left(\frac{0.45 - \bar{C}}{0.45}\right)$$
$$\text{If } \max(D_{\text{labeled}}) > 0.70 \implies w_H \leftarrow w_H + 0.05$$

$$\text{Unified Score } S(x) = w_U \cdot \bar{U}(x) + w_D \cdot \bar{D}_{\text{labeled}}(x) + w_H \cdot \bar{H}(x)$$

---

## 3. Seed 42 Screening Benchmark Results

Evaluated across Indian Pines, Salinas, and Pavia University under identical conditions:
- **Backbone**: 100% Frozen `spec_resnet50` (0 trainable backbone parameters).
- **Classifier**: 2-layer MLP (524K parameters, 25 epochs per round, lr=0.0005, batch size 64).
- **Pool Split**: 10 samples/class initial pool, $k=20$ queried per round over 5 rounds.

### Screening Summary Table (Seed 42):

| Dataset | Strategy | Final OA (%) | Final AA (%) | Macro F1 | ECE | AULC_OA (%) | Query Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** | Random | 60.60% | 65.22% | 0.5490 | 0.0916 | 55.49% | 0.001s |
| **Indian Pines** | Least Confidence | 60.01% | 68.55% | 0.5929 | 0.0673 | 53.29% | 0.003s |
| **Indian Pines** | LC + Farthest Diversity | 59.36% | 68.03% | 0.5862 | **0.0562** | 53.55% | 0.042s |
| **Indian Pines** | **Adaptive Controller** | 59.82% | **68.70%** | 0.5775 | 0.0692 | 53.67% | 0.048s |
| | | | | | | | |
| **Salinas** | Random | 68.92% | 71.72% | 0.7043 | **0.0214** | 63.25% | 0.001s |
| **Salinas** | Least Confidence | 69.27% | 76.18% | 0.7312 | 0.0361 | 63.23% | 0.004s |
| **Salinas** | **LC + Farthest Diversity** | **72.28%** | **76.77%** | **0.7573** | 0.0676 | **65.11%** | 0.051s |
| **Salinas** | Adaptive Controller | 68.87% | 74.95% | 0.7281 | 0.0405 | 63.25% | 0.059s |
| | | | | | | | |
| **Pavia University** | **Random** | **62.68%** | 62.55% | 0.5686 | **0.1822** | **56.68%** | 0.001s |
| **Pavia University** | Least Confidence | 53.95% | 62.88% | 0.5628 | 0.2838 | 49.81% | 0.005s |
| **Pavia University** | **LC + Farthest Diversity** | 58.89% | **64.62%** | **0.6036** | 0.2355 | 52.99% | 0.046s |
| **Pavia University** | Adaptive Controller | 55.26% | 63.11% | 0.5673 | 0.2555 | 50.43% | 0.052s |

---

## 4. Per-Round Controller Weight Progression

Tracked dynamically during the active learning iterations:

```csv
dataset,seed,round,w_uncertainty,w_diversity,w_hard_region
Indian Pines,42,0,0.2409,0.5985,0.1606
Indian Pines,42,1,0.3212,0.5182,0.1606
Indian Pines,42,2,0.4015,0.4379,0.1606
Indian Pines,42,3,0.4818,0.3576,0.1606
Indian Pines,42,4,0.5621,0.2773,0.1606
```

- **Early Rounds ($r=0$)**: Diversity weight $w_D = 0.60$ dominates to sample representative feature regions.
- **Late Rounds ($r=4$)**: Uncertainty weight $w_U = 0.56$ dominates to target class boundaries.

---

## 5. Web Visualization Application Integration

The `AdaptiveQueryController` logic and weight reporting have been fully integrated into `src/visualization_app/engine_wrapper.py`:
- In the Web UI dashboard, selecting `Adaptive Query Controller` displays the exact per-round weights ($w_U, w_D, w_H$) alongside candidate rank explanations.
- Sample explanation output format:
  > *"Adaptive Controller Rank #1: Uncertainty w_U=0.24 (LC: 0.837, top 0.5%), Diversity w_D=0.60 (Score: 0.812, top 1.2%), Hard-Region w_H=0.16."*

---

## 6. System Verification Checklist

- [x] Backbone 100% frozen (`requires_grad = False`, 0 trainable backbone parameters).
- [x] No test data leakage in controller decisions.
- [x] Pre-extracted feature caching preserved.
- [x] Baseline results (`Random`, `Least Confidence`, `LC + Farthest Diversity`) preserved.
- [x] Per-round controller weight progression logged in `logs/al_experiments/controller_weights.csv`.
- [x] Summary metrics exported to `logs/al_experiments/controller_screening_summary.csv`.
