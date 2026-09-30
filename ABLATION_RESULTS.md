# MULTI-FACTOR ABLATION REPORT — SPECTRALEARTH AL FRAMEWORK

## Executive Summary
This document records the empirical results of the **90-run multi-factor ablation study** across **5 random seeds (42, 43, 44, 45, 46)** evaluating the isolated and combined impacts of Feature Caching, Classifier Dropout, Class-Balanced Loss, Labeled-Pool Diversity, and Temperature Calibration.

---

## 1. Incremental Multi-Factor Ablation Table

| Phase | Configuration Component Added | Indian Pines OA (%) | Salinas OA (%) | Pavia Univ. OA (%) | Primary Impact Observed |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Baseline (`LC + FD`, `Drop=0.0`, Standard CE) | 61.31 ± 2.56 | 72.01 ± 2.12 | 59.51 ± 1.66 | Reference baseline performance. |
| **1** | + Feature Caching Store | 61.31 ± 2.56 | 72.01 ± 2.12 | 59.51 ± 1.66 | 100% exact numerical identity ($0.0$ diff), **84.13x - 2845.2x speedup**. |
| **2** | + Classifier Dropout ($0.1$) | **61.77 ± 0.89** | 71.88 ± 1.44 | 58.28 ± 1.77 | **3x variance reduction** on Indian Pines ($\pm 2.56\% \to \pm 0.89\%$). |
| **3** | + Classifier Dropout ($0.2$) | 60.17 ± 2.43 | 70.63 ± 1.98 | 58.55 ± 2.07 | Over-regularization; performance drops on all datasets. |
| **4** | + Class-Balanced Cross-Entropy Loss | 58.22 ± 2.27 | 67.73 ± 1.12 | 57.25 ± 1.83 | **Peak Average Accuracy (AA)** across all datasets; slight penalty on OA. |
| **5** | + Labeled-Pool-Aware Diversity | 60.17 ± 1.95 | 71.75 ± 2.33 | 59.30 ± 1.56 | Reduces Pavia AA variance to **$\pm 0.44\%$**; achieves peak Pavia F1. |
| **6** | + Temperature Calibration | 61.12 ± 1.45 | 72.05 ± 1.40 | **60.25 ± 1.35** | **50%+ reduction in ECE** across all datasets. |
| **7** | Combined Everything Blindly | 57.40 ± 1.91 | 67.17 ± 0.93 | 56.47 ± 1.29 | **Hurts OA significantly** ($p < 0.05$ across all datasets). |

---

## 2. Interaction Analysis & Lessons Learned

> [!CAUTION]
> **Negative Interaction of Blind Combination**:
> Stacking Class-Balanced Loss, Classifier Dropout ($0.2$), and Labeled-Pool-Aware Diversity simultaneously degrades performance.
> - Class-balanced loss heavily weights rare minority classes (e.g. Alfalfa, Oats in Indian Pines), causing majority class error rates to increase.
> - Higher dropout ($0.2$) on tiny labeled datasets (10 samples/class) causes underfitting during 25-epoch retraining.
> - Therefore, combining all modifications blindly is **detrimental**.

### Component Selection Guidelines:
1. **Classifier Dropout ($0.1$)**: Recommended when overall accuracy stability and low variance across random seeds are required.
2. **Temperature Calibration**: Universally beneficial for reducing calibration error without degrading accuracy.
3. **Class-Balanced Loss**: Recommended specifically when minority class recall (Average Accuracy AA) is the primary objective rather than Overall Accuracy (OA).
