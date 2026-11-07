# ConformAL-HSI: Historical Experiment Reconciliation & Lineage

**Document:** Formal Historical Reconciliation & Lineage Analysis  
**Date:** November 2026  
**Status:** Validated  

This document explains the origins of all historical numerical variations across the repository's research trajectory, resolving conflicting reports and establishing strict comparability boundaries.

---

## 1. Resolution of the Historical $N=220$ Discrepancies

Across earlier project reports, conflicting Overall Accuracy numbers were cited for identical dataset names at label budget $N=220$. Forensic source code and parameter auditing reveals the exact cause of every variation:

| Historical Study | Commit | Head Architecture | Training Setup | Salinas OA at $N=220$ | Indian Pines OA at $N=220$ | Primary Reason for Numerical Difference |
|:---|:---:|:---|:---|:---:|:---:|:---|
| **Phase 1–3 Adaptive Optimization** | `a62cb95` | **Head B** (512 units, BatchNorm, Dropout 0.1, input L2-norm) | AdamW, $\text{lr}=2 \times 10^{-3}$, 45 epochs, CosineAnnealing | **74.60%** (Margin+Cos) | **65.10%** (Margin+Cos) | High-capacity regularized head with BatchNorm feature scaling; evaluated across 3 seeds (42–44). |
| **Controlled Three-Way Comparison** | `7c15894` | **Head A** (256 units, LayerNorm(2048), LayerNorm(256), GELU) | AdamW, $\text{lr}=5 \times 10^{-4}$, 15 epochs, CosineAnnealing | **61.34%** (Margin+Cos) | **51.59%** (Margin+Cos) | Under-parameterized 256-unit head without dropout; evaluated across 5 seeds (42–46). Lower capacity leads to boundary underfitting. |
| **Capacity Ablation Study** | `25d3f7c` | **Head A (Bare)** (256 units, NO Normalization, ReLU) | AdamW, $\text{lr}=2 \times 10^{-3}$, 45 epochs | **32.14%** (Margin+Cos) | **30.21%** (Margin+Cos) | **Pathological Configuration:** Omission of input/hidden LayerNorm combined with high learning rate ($2 \times 10^{-3}$) on unnormalized 2048-dim features caused severe optimization divergence. |
| **Extended Budget Study** | `d900587` | **Head B** (512 units, BatchNorm, Dropout 0.1) | AdamW, $\text{lr}=2 \times 10^{-3}$, 45 epochs | **75.78%** (Margin+Cos) | **61.78%** (Margin+Cos) | Head B with 5 seeds (42–46) and $k=200$ sample jumps from $N=220 \to 2000$. |
| **Master Consolidation Benchmark** | Current | **Head A (Standardized)** (256 units, LayerNorm) | AdamW, $\text{lr}=5 \times 10^{-4}$, 15 epochs | **59.68%** (Margin+Cos) | **50.64%** (Margin+Cos) | Strict Head A benchmark comparing 7 strategies with split-conformal calibration on validation pool. |

---

## 2. Classification of Historical Experiments

Every major experimental suite in `results/` is classified into one of four reproducibility and comparability categories:

```
[Directly Comparable - Head B Cohort]
  ├── results/label_efficiency/ (Exp 1 Unified Multi-Budget)
  ├── results/track_b/ (External Model Benchmarking)
  ├── results/label_efficiency_optimization/ (Phases 1-3 Advanced AL)
  └── results/extended_label_budget_study/ (N=220 to 2000 Scaling)

[Directly Comparable - Head A Cohort]
  ├── results/original_vs_updated_active_learning/ (Three-Way AL Comparison)
  └── results/master_research_consolidation/ (Master Conformal AL Suite)

[Cross-Architecture Diagnostic (Restricted Comparability)]
  └── results/final_research_validation/ (Capacity Ablation Study)

[Superseded / Invalid for Primary Benchmarking]
  └── Capacity Ablation Head A_Bare (diverged due to missing normalization and lr=2e-3)
```

---

## 3. Discrepancy Between Seed 42 and Multi-Seed Ensemble Crossings

Early project notes reported single-seed (Seed 42) threshold crossings:
- Indian Pines reached 60% OA at $N=360$ (Seed 42 OA = 61.05%).
- Salinas reached 60% OA at $N=240$ (Seed 42 OA = 61.27%).
- Pavia reached 60% OA at $N=250$ (Seed 42 OA = 60.28%).

In contrast, the 5-seed aggregated report stated:
- Indian Pines: Mean curve never reaches 60% (peaking at $57.30\%$).
- Salinas: Mean curve reaches 60% at **$N=220$** ($61.34\%$).
- Pavia: Mean curve reaches 60% at **$N=290$** ($60.44\%$).

**Forensic Finding:** These differences represent standard statistical differences between a single seed trajectory and the 5-seed ensemble mean curve:
- On Salinas, Seeds 44, 45, and 46 crossed at $N=180–200$, accelerating the ensemble mean crossing to $N=220$ despite Seed 42 crossing at $N=240$.
- On Indian Pines, Seeds 43, 44, and 46 experienced lower accuracy, dragging the ensemble mean below 60%.
Both numbers are verified as mathematically genuine. The error was descriptive conflation in early summaries.
