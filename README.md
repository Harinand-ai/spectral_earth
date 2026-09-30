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
