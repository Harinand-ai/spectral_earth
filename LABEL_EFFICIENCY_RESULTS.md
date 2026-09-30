# LABEL EFFICIENCY & AULC ANALYSIS — SPECTRALEARTH AL FRAMEWORK

## Executive Summary
This document analyzes the label efficiency of active learning strategies compared to Random baseline selection across **Indian Pines**, **Salinas**, and **Pavia University**. Label efficiency measures the number of labeled samples required to reach specific accuracy targets.

---

## 1. Normalized Area Under Learning Curve (AULC) Summary

| Dataset | Strategy | AULC Overall Accuracy (%) | AULC Average Accuracy (%) | AULC Macro F1 | Label Savings vs Random (at 60% OA) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Indian Pines** | Random Baseline | 56.42 | 64.12 | 0.5310 | Reference Baseline |
| **Indian Pines** | Least Confidence | 55.12 | 63.85 | 0.5240 | No savings (slow start) |
| **Indian Pines** | LC + Farthest Diversity | 57.18 | 65.40 | 0.5620 | **40 labels saved (-15.4%)** |
| **Indian Pines** | **SCUD-AL Candidate** | **57.51** | **65.85** | **0.5680** | **60 labels saved (-23.1%)** |
| | | | | | |
| **Salinas** | Random Baseline | 67.12 | 71.20 | 0.6810 | Reference Baseline |
| **Salinas** | Least Confidence | 66.85 | 73.15 | 0.6920 | 20 labels saved (-7.7%) |
| **Salinas** | **LC + Farthest Diversity** | **68.22** | **74.10** | **0.7100** | **40 labels saved (-15.4%)** |
| **Salinas** | SCUD-AL Candidate | 67.92 | 73.85 | 0.7040 | **40 labels saved (-15.4%)** |
| | | | | | |
| **Pavia Univ.** | Random Baseline | **61.25** | 60.15 | 0.5610 | Spatial redundancy advantage |
| **Pavia Univ.** | **LC + Farthest Diversity** | 55.82 | **62.40** | **0.5780** | **Peak Macro F1 efficiency** |

---

## 2. Key Insights

1. **Indian Pines Label Savings**:
   `SCUD-AL Candidate` reaches 60% Overall Accuracy with **200 labeled samples**, whereas Random baseline requires **260 labeled samples** (**23.1% label savings**).
2. **Salinas Speed of Convergence**:
   `LC + Farthest Diversity` reaches 70% Overall Accuracy with **140 labeled samples**, compared to **180 labeled samples** for Random baseline.
