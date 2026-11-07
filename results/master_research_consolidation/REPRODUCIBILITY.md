# ConformAL-HSI: Master Reproducibility Guide

**Document:** Complete Reproducibility & Replication Guide  
**Date:** November 2026  
**Status:** Validated  

---

## 1. System & Environment Specifications

- **Operating System:** Windows 11 (PowerShell terminal).
- **Python Version:** Python 3.11+.
- **PyTorch / CUDA:** PyTorch `2.11.0+cu128`, CUDA Available, GPU: `NVIDIA GeForce RTX 2050` (4 GB VRAM).
- **Required Libraries:** `torch`, `scipy`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`.

---

## 2. Directory Structure & Cached Features

All experiments utilize pre-extracted, frozen 2048-dimensional feature representations from the SpectralEarth 3D-ResNet50 backbone:
```
cache/preextracted_features/
├── indian_pines_train_feats.pt       (7174 samples, 2048-dim)
├── indian_pines_val_feats.pt         (1537 samples, 2048-dim)
├── indian_pines_test_feats.pt        (1538 samples, 2048-dim)
├── salinas_train_feats.pt            (37890 samples, 2048-dim)
├── salinas_val_feats.pt              (8119 samples, 2048-dim)
├── salinas_test_feats.pt             (8120 samples, 2048-dim)
├── pavia_university_train_feats.pt   (29943 samples, 2048-dim)
├── pavia_university_val_feats.pt     (6416 samples, 2048-dim)
└── pavia_university_test_feats.pt    (6417 samples, 2048-dim)
```

---

## 3. Step-by-Step Reproduction Instructions

### Step 1: Run Automated Test Suite
To verify the Split Conformal Prediction engine, nonconformity quantiles, coverage guarantees, and class-coverage constraints:
```bash
.\venv\Scripts\python.exe tests/test_conformal_al.py
```
*Expected Output:* `Ran 5 tests in ~0.15s: OK`.

### Step 2: Reproduce Spatial Leakage Quantification
To quantify the exact patch overlap and spatial proximity of the random pixel split:
```bash
.\venv\Scripts\python.exe scratch/audit_spatial_leakage.py
```
*Expected Output:* 100.00% patch overlap; mean Euclidean distance 1.00–1.01 pixels.

### Step 3: Reproduce Controlled Classifier Head Standardization
To run the 4-head ablation (Head A 256-LN vs Head B 512-BN-Drop vs Head B-LN 512-LN-Drop vs Linear Probe):
```bash
.\venv\Scripts\python.exe scratch/run_controlled_head_standardization.py
```
*Output File:* `scratch/controlled_head_standardization_results.csv`.

### Step 4: Run Master Conformal Active Learning Benchmark
To execute the 1,155 checkpoint evaluations across all 7 strategies, 5 seeds ($42, 43, 44, 45, 46$), and 3 datasets:
```bash
.\venv\Scripts\python.exe run_conformal_active_learning_experiment.py
```
*Output Files:*
- `results/master_research_consolidation/conformal_al_seed_results.csv`
- `results/master_research_consolidation/conformal_al_per_class_metrics.csv`
- `results/master_research_consolidation/conformal_al_diagnostics.csv`

### Step 5: Compute Independent Seed-Level Statistics (AUCL)
To calculate independent Area Under Learning Curve metrics and seed-level paired $t$-tests:
```bash
.\venv\Scripts\python.exe scratch/compute_seed_level_statistics.py
```
*Output Files:*
- `results/master_research_consolidation/seed_level_aucl_summary.csv`
- `results/master_research_consolidation/seed_level_paired_statistical_analysis.csv`

### Step 6: Compute Label Efficiency Milestones & Generate Figures
To compute discrete label milestones and produce 300 DPI publication plots:
```bash
.\venv\Scripts\python.exe scratch/compute_master_label_efficiency.py
```
*Output Files:*
- `results/master_research_consolidation/master_aggregated_results.csv`
- `results/master_research_consolidation/master_label_efficiency_milestones.csv`
- `results/master_research_consolidation/master_three_way_comparison_master.png`
- `results/master_research_consolidation/conformal_coverage_and_set_size.png`
