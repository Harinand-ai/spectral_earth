# Cross-Sensor & Cross-Dataset Transfer Evaluation — Scientific Audit Report

This document records the official **Scientific Audit** of the cross-sensor / cross-dataset transfer experiments for the **SpectralEarth Active Learning Framework**.

---

## 1. Audit Scope & Executive Summary

- **Trigger**: The initial run of `run_cross_sensor_eval.py` produced identical results for `Indian Pines -> Pavia University` and `Salinas -> Pavia University` (**61.60% OA, 66.42% AA, 0.6226 F1, 0.5176 Kappa**).
- **Audit Findings**:
  - **Root Cause Identified**: The initial script loop for Experiment 2 iterated over `(src_name, tgt_name)` but failed to invoke source training or transfer source weights to `tgt_engine`. It executed plain within-dataset Active Learning directly on `Pavia University` under `seed = 42` for both transfer pairs.
  - **Correction Implemented**: `run_cross_sensor_eval.py` was updated to train the source classifier head on `src_name`, verify source FC1 weight signatures, copy hidden layer parameters to `tgt_engine.classifier_head`, and pass target patch tensors through `SpectralResampler1D`.
  - **Verified Independent Metrics**:
    - `Indian Pines -> Pavia University`: **49.84% OA | 53.07% AA | 0.4883 F1 | 0.3842 Kappa** (Source Signature: `-87.374374`).
    - `Salinas -> Pavia University`: **54.12% OA | 55.22% AA | 0.5190 F1 | 0.4261 Kappa** (Source Signature: `-69.446945`).
    - `Indian Pines -> Salinas`: **74.14% OA | 80.07% AA | 0.7781 F1 | 0.7147 Kappa** (Source Signature: `-87.374374`).
    - `Salinas -> Indian Pines`: **58.26% OA | 67.32% AA | 0.5801 F1 | 0.5271 Kappa** (Source Signature: `-69.446945`).
- **Core Architecture Integrity**: The frozen SpectralEarth backbone (`spec_resnet50`), Model A, Model B, and all within-dataset active learning results are **100% UNTOUCHED AND REPRODUCIBLE**.

---

## 2. Code Trace & Root Cause Analysis

### Uncorrected Loop Defect
In the uncorrected `run_cross_sensor_eval.py`, the loop:
```python
for src_name, tgt_name, pair_desc in transfer_pairs:
    tgt_dm = get_datamodule(tgt_name)
    tgt_engine = LearningEngine(...)
    history = tgt_engine.run_full_active_learning_loop(...)
```
never referenced `src_name`. As a result, when evaluating `Indian Pines -> Pavia University` and `Salinas -> Pavia University`, both iterations simply initialized a fresh target model on Pavia University under `seed = 42` and ran standard active learning, producing duplicate outputs.

### Corrected Pipeline Execution
In the corrected implementation:
1. `src_engine` is trained on `src_name` labeled pool $L_{\text{source}}$.
2. The weight sum signature of `src_engine.classifier_head.head[0].weight` is logged.
3. The FC1 weights and bias are explicitly transferred:
   `tgt_engine.classifier_head.head[0].weight.data.copy_(src_engine.classifier_head.head[0].weight.data)`
4. `SpectralResampler1D(target_bands=src_bands)` resamples target input patches before feature extraction.
5. Few-Shot target active learning adaptation ($LC+FD$) is executed on the target labeled pool.

---

## 3. Class-Space & Semantic Analysis

| Property | Indian Pines | Salinas | Pavia University |
| :--- | :--- | :--- | :--- |
| **Classes ($C$)** | 16 | 16 | 9 |
| **Land-Cover Domain** | Agriculture (Indiana, USA) | Agriculture (California, USA) | Urban / Built-up (Pavia, Italy) |
| **Sample Classes** | Alfalfa, Corn, Soybean | Brocoli, Celery, Vinyard | Asphalt, Meadows, Bricks |

- **Zero-Shot Classification Direct Output Layer Reuse**: **INVALID — SCIENTIFICALLY INCOMPATIBLE** for direct land-cover classification. Class 1 in Indian Pines represents *Alfalfa*; Class 1 in Salinas represents *Brocoli green weeds*. Without target class output layer re-alignment, direct zero-shot predictions yield near-random accuracy ($15.25\%$ and $5.59\%$).
- **Few-Shot Active Learning Target Adaptation**: **VALID WITH LIMITATIONS**. Transferring pre-trained hidden feature representations ($W_{\text{fc1}} \in \mathbb{R}^{256 \times 2048}$) provides a strong feature projection space, allowing the output classification layer ($W_{\text{out}} \in \mathbb{R}^{C_{\text{tgt}} \times 256}$) to adapt rapidly to target classes using target active learning queries.

---

## 4. Final Scientific Decision Matrix

| Transfer Pair | Mode | Audit Decision | Metric Summary |
| :--- | :---: | :---: | :--- |
| **Indian Pines $\to$ Salinas** | Zero-Shot Transfer | **INVALID — SCIENTIFICALLY INCOMPATIBLE** | 15.25% OA (Semantic label shift) |
| **Salinas $\to$ Indian Pines** | Zero-Shot Transfer | **INVALID — SCIENTIFICALLY INCOMPATIBLE** | 5.59% OA (Semantic label shift) |
| **Indian Pines $\to$ Salinas** | Few-Shot AL Adaptation | **VALID WITH LIMITATIONS** | **74.14% OA \| 0.7781 F1** |
| **Salinas $\to$ Indian Pines** | Few-Shot AL Adaptation | **VALID WITH LIMITATIONS** | **58.26% OA \| 0.5801 F1** |
| **Indian Pines $\to$ Pavia Univ.** | Few-Shot AL Adaptation | **VALID WITH LIMITATIONS** | **49.84% OA \| 0.4883 F1** |
| **Salinas $\to$ Pavia Univ.** | Few-Shot AL Adaptation | **VALID WITH LIMITATIONS** | **54.12% OA \| 0.5190 F1** |

---

## 5. Verification Commands

```powershell
d:\Projects\spectral_earth\venv\Scripts\python.exe d:\Projects\spectral_earth\run_cross_sensor_eval.py
```
