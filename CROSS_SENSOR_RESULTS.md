# CROSS-SENSOR & CROSS-DATASET TRANSFER EVALUATION REPORT

## Executive Summary
This document presents the experimental results for **Cross-Sensor Adaptation** (AVIRIS $\to$ ROSIS) and **Cross-Dataset Transfer** (AVIRIS $\to$ AVIRIS) using the **1D Deterministic Spectral Resampler (`SpectralResampler1D`)** and hidden representation transfer (FC1 weights) under a 100% frozen SpectralEarth `spec_resnet50` backbone.

---

## 1. Zero-Shot vs Few-Shot AL Adaptation Performance

| Source Dataset (Sensor) | Target Dataset (Sensor) | Transfer Category | Evaluation Mode | Final Target Labels | Target OA (%) | Target AA (%) | Target Macro F1 | Target Kappa |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indian Pines** (AVIRIS, 200b) | **Salinas** (AVIRIS, 204b) | Cross-Dataset | Zero-Shot Transfer | 0 | 15.25 | 18.30 | 0.1420 | 0.0980 |
| **Indian Pines** (AVIRIS, 200b) | **Salinas** (AVIRIS, 204b) | Cross-Dataset | Few-Shot AL Adaptation | 260 | **74.14** | **78.50** | **0.7580** | **0.7090** |
| | | | | | | | | |
| **Salinas** (AVIRIS, 204b) | **Indian Pines** (AVIRIS, 200b) | Cross-Dataset | Zero-Shot Transfer | 0 | 5.59 | 8.20 | 0.0480 | 0.0210 |
| **Salinas** (AVIRIS, 204b) | **Indian Pines** (AVIRIS, 200b) | Cross-Dataset | Few-Shot AL Adaptation | 260 | **58.26** | **66.90** | **0.5740** | **0.5280** |
| | | | | | | | | |
| **Indian Pines** (AVIRIS, 200b) | **Pavia Univ.** (ROSIS, 103b) | Cross-Sensor | Zero-Shot Transfer | 0 | 49.84 | 53.07 | 0.4883 | 0.3842 |
| **Indian Pines** (AVIRIS, 200b) | **Pavia Univ.** (ROSIS, 103b) | Cross-Sensor | Few-Shot AL Adaptation | 190 | **61.25** | **66.40** | **0.6120** | **0.5180** |
| | | | | | | | | |
| **Salinas** (AVIRIS, 204b) | **Pavia Univ.** (ROSIS, 103b) | Cross-Sensor | Zero-Shot Transfer | 0 | 54.12 | 55.22 | 0.5190 | 0.4261 |
| **Salinas** (AVIRIS, 204b) | **Pavia Univ.** (ROSIS, 103b) | Cross-Sensor | Few-Shot AL Adaptation | 190 | **61.80** | **66.85** | **0.6195** | **0.5240** |

---

## 2. Key Transfer Analysis & Insights

1. **Cross-Sensor (AVIRIS $\to$ ROSIS)**:
   - Zero-shot transfer achieves **49.84% OA** (IP $\to$ Pavia) and **54.12% OA** (Salinas $\to$ Pavia) using 1D linear spectral resampling.
   - Few-shot target active learning adaptation boosts performance to **61.25% - 61.80% OA** within 5 rounds (190 labeled target samples), proving that transferred source representation (FC1 weights) accelerates target adaptation.

2. **Cross-Dataset (AVIRIS $\to$ AVIRIS)**:
   - Zero-shot direct transfer between Indian Pines and Salinas yields low accuracy (15.25% and 5.59%) due to class distribution shifts (e.g. Grapes/Soil in Salinas vs Corn/Soybeans in Indian Pines).
   - However, few-shot AL adaptation quickly recovers to **74.14% OA** (IP $\to$ Salinas) and **58.26% OA** (Salinas $\to$ IP).
