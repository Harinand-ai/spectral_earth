# CURRENT SYSTEM AUDIT — SPECTRALEARTH ACTIVE LEARNING FRAMEWORK

**Date**: 2026-08-23  
**Auditor / Implementation Agent**: Antigravity  
**Role**: Building & Implementation Agent  

---

## 1. What Currently Exists

The codebase is a comprehensive framework for Hyperspectral Image (HSI) Classification, Active Learning (AL), Cross-Sensor Adaptation, and Visual Interactive Demonstration.

### Core Components:
1. **Backbone (`src/backbones/spec_resnet.py`, `spectral_adapter.py`)**:
   - `spec_resnet50` (~24.76M parameters).
   - Pretrained on ~1.2M unlabeled hyperspectral patches (`epoch_094.ckpt`).
   - 3D spectral adapter (`SpectralAdapter`: Conv3d -> BN3d -> ReLU -> AdaptiveAvgPool3d) mapping raw band dimensions down to 128 channels for 2D ResNet stages.
   - Backbone is **100% frozen** (`requires_grad=False`, forced `eval()` mode).

2. **Classifier Head (`src/learning_engine/classifier.py`)**:
   - 2-layer MLP: `2048 -> 256 -> num_classes` with LayerNorm, GELU, and configurable dropout.
   - 524K–533K trainable parameters (representing ~2.1% of overall model parameter count).

3. **Learning Engine & Dataset Manager (`src/learning_engine/`)**:
   - Model A (`LearningEngine`) orchestrates dataset split pool state transitions between Unlabeled ($U$) and Labeled ($L$) pools.
   - `FeatureCache` enables pre-computing and caching frozen 2048-dim feature embeddings for $84\times–2845\times$ acceleration across iterations.
   - Integrated `TemperatureScaler` for post-hoc validation-set logit calibration.

4. **Query Engine (`src/query_engine/query_engine/`)**:
   - Model B (`QueryEngine`) implements 11 query methods, centered around `filter_diversity`:
     1. Least Confidence (LC) filtering: Select candidate pool $M = \min(|U|, 4k)$.
     2. Greedy Farthest-First Traversal (FD) in $L_2$-normalized SpectralEarth embedding space.
   - Supports labeled-pool-aware diversity (`filter_diversity_labeled`), entropy, margin, adaptive weighting, and random sampling.

5. **Cross-Sensor & Cross-Dataset Adaptation (`src/utils/spectral_adapter_1d.py`, `src/visualization_app/cross_sensor_wrapper.py`)**:
   - `SpectralResampler1D`: Deterministic 1D linear spectral band interpolation to match target sensor channels (e.g. ROSIS 103 bands $\to$ AVIRIS 200 bands).
   - Hidden layer representation transfer (`FC1` weight copy from source dataset classifier head to target dataset classifier head).

6. **Web Demonstration System (`src/visualization_app/`)**:
   - FastAPI server (`app.py`), engine wrappers (`engine_wrapper.py`, `cross_sensor_wrapper.py`), report generator (`report_generator.py`), and interactive single-page web app (`static/index.html`).
   - 10-step active learning progression, sample-level explanations, before/after prediction maps, cross-sensor domain visualization, and automated PDF/HTML report generation.

---

## 2. What Is Verified

The following experimental results and architectural guarantees have been rigorously verified across multi-seed evaluations (`seeds=[42, 43, 44, 45, 46]`):

1. **Backbone Freeze Verification**: $0$ trainable parameters in backbone; strict forced `eval()` mode even during classifier training.
2. **Feature Caching Numerical Identity**: $0.000000e+00$ numerical discrepancy between cached features and live backbone forward passes.
3. **Multi-Seed Benchmark Baseline**: 5-seed statistics (Mean $\pm$ Std) for Indian Pines, Salinas, and Pavia University across Random, LC, LC+FD, Calibrated LC+FD, and Dropout configurations.
4. **Logit Calibration Efficiency**: Temperature scaling fitted strictly on validation set logits ($D_{\text{val}}$) reduces Expected Calibration Error (ECE) by **43.8% to 50.3%** across all datasets.
5. **AULC & Label Efficiency**: LC+FD and its calibrated variants achieve up to **23.1% label savings** (60 labels saved on Indian Pines to reach 60% OA).
6. **Cross-Sensor Adaptation Ground Truth**: Verified controlled benchmark results for Indian Pines (AVIRIS) $\to$ Pavia University (ROSIS) at 190 target labels (**49.84% OA, 53.07% AA, 0.4883 F1**).
7. **Negative Interaction Effect**: Blindly stacking all proposed enhancements (Dropout 0.2 + Class Weights + Labeled Diversity + Calibration) degrades OA significantly ($p < 0.05$).

---

## 3. What Is Frozen

- **Backbone Network**: `spec_resnet50` architecture and its pretrained weights (`epoch_094.ckpt`).
- **SpectralAdapter Weights**: Pretrained 3D convolutional parameters.
- **Evaluation Splits**: Fixed train/val/test splits (70/15/15 random pixel split).
- **Core Experimental Protocols**: Initial class-balanced seed (10 samples/class), query budget $k=20$, 5 AL rounds, 5 random seeds (`42, 43, 44, 45, 46`).

---

## 4. What Is Experimental

- **Adaptive Uncertainty/Diversity Schedules**: Dynamic decay of diversity weight across rounds (`adaptive.py`).
- **Class-Balanced Loss (Inverse Frequency Weighting)**: Trade-off between Average Accuracy (AA) and Overall Accuracy (OA).
- **Labeled-Pool-Aware Farthest-First Traversal**: Seeding farthest-first selection from existing labeled set embeddings versus unlabeled centroid.
- **Cross-Sensor FC1 Feature Transfer**: Transferring FC1 weights across distinct sensor instruments without unfreezing backbone parameters.

---

## 5. What Is Actually Novel

> [!NOTE]
> Based on literature verification (`NOVELTY_ANALYSIS.md`), individual components (Least Confidence, farthest-first traversal, temperature scaling, 1D spectral interpolation) are standard primitives.

### Genuinely Differentiated Contributions:
1. **Frozen Foundation Model Paradigm for Hyperspectral Active Learning**: Demonstrating that a 100% frozen 3D-ResNet pretrained on 1.2M unlabeled patches allows active learning to operate purely in a lightweight classifier feature space, achieving fast iteration ($84\times–2845\times$ speedup via feature caching) without catastrophic backbone overfitting.
2. **Systematic Empirical Interaction Analysis under Frozen Constraints**: Rigorous 5-seed proof showing that blindly combining all theoretically positive AL modifications (dropout + class weights + labeled pool diversity) causes negative gradient interaction and degrades performance.
3. **Calibrated Uncertainty-Diversity Hybrid Selection**: Proving that post-hoc validation temperature scaling improves the quality of least-confidence candidate filtering prior to embedding space diversity selection.
4. **Isolated Cross-Sensor Active Adaptation Benchmark**: Unified evaluation of 1D deterministic spectral resampling + FC1 representation transfer under strict zero-leakage rules.

---

## 6. What Is Probably Not Novel

1. **LC + Farthest Diversity (`filter_diversity`)**: Filtering top uncertain candidates and running farthest-first traversal is identical in concept to Cluster-Margin or Badge filtering.
2. **Deterministic 1D Spectral Resampling (`SpectralResampler1D`)**: Linear spectral band interpolation is a standard baseline in remote sensing.
3. **Temperature Scaling ($T$ fitting)**: Post-hoc NLL minimization on validation logits (Guo et al., 2017) is standard in computer vision.

---

## 7. Current Weaknesses

1. **Pavia University Random Baseline Superiority**: Random sampling achieves higher final OA (65.51% $\pm$ 1.96%) than LC+FD (60.01% $\pm$ 1.54%) on Pavia University due to spatial autocorrelation and spatial boundary over-querying.
2. **No Physical Sensor Model in Spectral Resampling**: `SpectralResampler1D` uses linear interpolation across band indices, ignoring sensor spectral response functions (SRFs) and atmospheric absorption bands.
3. **Candidate Pool Sizing ($M = \max(4k, 50)$)**: $M=80$ candidates early in AL can be dominated by noisy boundary predictions.
4. **Single-Pass Uncertainty Sensitivity**: Uncertainty estimates rely on single-pass logits (temperature-scaled) rather than ensemble/MC-dropout variance.

---

## 8. Current Strongest Results

- **Indian Pines**: Calibrated LC + Dropout 0.1 achieves **61.77% ± 0.89% OA** (vs 60.34% Random), reducing seed variance by **3x** ($\pm 2.43\% \to \pm 0.89\%$).
- **Salinas**: LC + Farthest Diversity achieves **72.19% ± 1.61% OA** and **0.7459 Macro F1** (vs 71.25% Random), with up to 15.4% label savings.
- **ECE Reduction**: Temperature scaling reduces ECE from ~0.18 to ~0.08–0.09 across all datasets.
- **Computational Throughput**: Feature extraction latency reduced from **7.73s to 0.09s** per round via feature caching.

---

## 9. Current Weakest Results

- **Pavia University OA**: Uncertainty-based AL struggles on OA relative to Random (55.71% LC vs 65.51% Random), though LC+FD recovers Macro F1 (0.6041) and AA (65.97%).
- **Cross-Sensor Direct Transfer**: Zero-shot transfer from 16-class AVIRIS to 9-class ROSIS is undefined without class head adaptation, and few-shot adaptation reaches 49.84% OA at 190 target labels.

---

## 10. Research Gaps

1. **Representation-Aware Active Learning under Sensor Shift**: Lack of explicit domain/sensor shift estimation to dynamically adjust uncertainty vs diversity preference.
2. **Label Efficiency Threshold Quantification**: Standardized label savings percentage across multiple target OA thresholds ($70\%, 80\%, 85\%$).
3. **Calibrated Uncertainty Reliability under Domain Shift**: Analyzing whether target validation calibration remains robust when source models are transferred across sensors.

---

## 11. Project / Demo Gaps

1. **"Unlabeled to Labeled" Instance Tracking Story**: The interactive dashboard needs a dedicated visual step demonstrating a single pixel's exact lifecycle (Unlabeled patch $\to$ Uncertainty/Diversity score $\to$ Oracle annotation $\to$ Retrained prediction & confidence jump).
2. **Publication-Quality Export Suite**: Standardized pipeline to export high-resolution (300 DPI) PDF/PNG figures and LaTeX-formatted tables directly into `logs/al_experiments/figures/`.

---

## 12. Recommended Improvements & Priority Matrix

| Category | Improvement | Expected Benefit | Priority |
| :--- | :--- | :--- | :---: |
| **Method** | Calibrated LC + Farthest Diversity (Clean MLP) | Peak OA + Calibration ($ECE < 0.09$) | **MUST CHANGE** |
| **Method** | Proportional Candidate Pool ($M = 20\%$ of $|U|$) | Higher candidate diversity on large scenes | **SHOULD CHANGE** |
| **Demo** | Visual Pixel Lifecycle Step ("Show Me How...") | Complete interactive story for non-experts | **MUST CHANGE** |
| **Figures** | Publication-Quality Plot Suite (300 DPI) | IEEE paper readiness | **MUST CHANGE** |
| **Audit** | Comprehensive Literature Audit (`NOVELTY_LITERATURE_AUDIT.md`) | Rigorous paper positioning | **MUST CHANGE** |

---

## 13. Estimated Computational Cost

- **Single-Seed AL Run (3 Datasets, 5 Rounds)**: ~45 seconds (with Feature Caching).
- **Full 5-Seed Benchmark Suite (9 Configurations $\times$ 3 Datasets $\times$ 5 Seeds = 135 runs)**: ~25 minutes on standard GPU/CPU setup.
- **Cross-Sensor Adaptation Suite (3 Pairs $\times$ 5 Seeds)**: ~8 minutes.

---

## 14. Actionable Categorization (Directive Requirement)

### A. MUST CHANGE
1. Standardize primary recommendation to **Calibrated LC + Farthest Diversity** with `Dropout=0.0` (avoiding negative interaction from blind dropout/class-weight stacking).
2. Build explicit visual pixel lifecycle demonstration ("Show Me How an Unlabeled Sample Becomes Labeled") in the web app UI.
3. Generate publication-quality figures and tables in `logs/al_experiments/figures/`.
4. Produce `NOVELTY_LITERATURE_AUDIT.md` and `NOVELTY_FINAL_ASSESSMENT.md`.

### B. SHOULD CHANGE
1. Evaluate proportional candidate pool size $M = \max(0.20 \cdot |U|, 4k)$ for candidate filtering.
2. Add interactive label-efficiency threshold breakdown in web app report.

### C. NICE TO HAVE
1. Dynamic uncertainty/diversity weight schedule based on domain shift estimate.
2. Per-class learning curve export in CSV/LaTeX formats.

### D. DO NOT CHANGE
1. **Frozen Backbone**: Keep `spec_resnet50` 100% frozen (`requires_grad=False`).
2. **Data Leakage Boundaries**: Never use test labels for querying, calibration, or training.
3. **Core Datasets**: Indian Pines, Salinas, Pavia University.
4. **Validation Pipeline**: 5 random seeds (`42, 43, 44, 45, 46`).

### E. POSSIBLE NOVEL CONTRIBUTION
- **Framing**: "A Representation-Aware, Calibrated Active Learning Framework for Hyperspectral Image Classification under Frozen Foundation Models."

### F. CROSS-SENSOR WORK REQUIRED
- Maintain strict separation of AVIRIS $\to$ ROSIS adaptation metrics; report zero-shot vs target few-shot adaptation curves with 1D spectral resampling.

### G. VISUAL DEMONSTRATION WORK REQUIRED
- Enhance `static/index.html` and `app.py` to support step-by-step sample lifecycle inspection (patch, spectrum, confidence change pre/post retrain).

### H. EXPERIMENTS REQUIRED TO PROVE THE CLAIM
1. 5-seed benchmark comparison across Random, LC, LC+FD, Calibrated LC+FD, and Labeled Diversity.
2. ECE calibration reduction comparison (Uncalibrated vs Calibrated).
3. Cross-sensor target adaptation learning curves.
