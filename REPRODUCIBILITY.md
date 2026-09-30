# REPRODUCIBILITY GUIDE — SPECTRALEARTH AL FRAMEWORK

## Overview
This document provides exact, step-by-step instructions to reproduce all experimental results, statistical tests, feature caching benchmarks, and figures within the **SpectralEarth Active Learning Framework**.

---

## 1. Environment & Prerequisites

- **OS**: Windows / Linux
- **Python**: 3.11+
- **PyTorch**: 2.11.0+cu128
- **Dependencies**: `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `pyrootutils`, `torchmetrics`

Activate virtual environment:
```bash
.\venv\Scripts\activate
```

---

## 2. Command Reference for Execution

### Phase A: Feature Caching Verification
Run exact numerical equivalence and speedup test:
```bash
.\venv\Scripts\python.exe scratch/verify_feature_cache.py
```
Output artifact: `logs/al_experiments/feature_cache_verification.json`

### Computational Profiling & Complexity
Profile parameters, FLOPs, and latency:
```bash
.\venv\Scripts\python.exe scratch/profile_computational_cost.py
```
Output artifact: `logs/al_experiments/COMPUTATIONAL_PROFILE.json`

### Master Active Learning Benchmark Pipeline (Phases B through N)
Run multi-seed (5 seeds: 42, 43, 44, 45, 46) benchmark suite across Indian Pines, Salinas, and Pavia University:
```bash
.\venv\Scripts\python.exe run_master_pipeline.py
```
Output artifacts:
- `logs/al_experiments/FINAL_MASTER_TABLE.csv`
- `logs/al_experiments/final_results.json`
- `logs/al_experiments/ablation_results.csv`
- `logs/al_experiments/calibration_results.csv`
- Figures in `logs/al_experiments/figures/`

### Cross-Sensor & Cross-Dataset Transfer Benchmark (Phases H & I)
Run AVIRIS $\to$ ROSIS and AVIRIS $\to$ AVIRIS transfer benchmark:
```bash
.\venv\Scripts\python.exe run_master_transfer.py
```
Output artifacts:
- `logs/al_experiments/cross_sensor_results.csv`

---

## 3. Seed Control & Random State Determinism

All random seeds are explicitly controlled via:
```python
torch.manual_seed(seed)
np.random.seed(seed)
```
Across all benchmarks, seeds `[42, 43, 44, 45, 46]` are identically applied to both baseline strategies and candidate methods to guarantee paired statistical significance testing.
