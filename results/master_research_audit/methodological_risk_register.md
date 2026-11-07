# ConformAL-HSI: Methodological Risk Register

**Module:** Master Research Audit & Reconciliation  
**Date:** November 2026  
**Status:** Validated & Forensic  

This register documents and ranks all methodological, scientific, statistical, and architectural risks identified during the forensic audit of the ConformAL-HSI codebase and historical experiments.

---

## Summary of Identified Risks

| Risk ID | Category | Severity | Title | Impact Summary |
|:---:|:---|:---:|:---|:---|
| **RISK-01** | Data Protocol | **CRITICAL** | Spatial Autocorrelation Leakage via Random Pixel Partitioning | Overlapping $15 \times 15$ patches across train and test sets inflate absolute accuracy. |
| **RISK-02** | Methodology | **CRITICAL** | Complete Absence of Conformal Guidance Implementation | Project title and claims promise conformal prediction, but no conformal pipeline exists in code. |
| **RISK-03** | Documentation | **HIGH** | Discrepancy Between Claimed (10/10/80) and Actual (70/15/15) Split | Prior audit reports claimed an 80% test set, but cached tensors enforce a 70% candidate / 15% test split. |
| **RISK-04** | Experimental | **HIGH** | Confounded Head A vs Head B Capacity Comparisons | Head A and Head B differed simultaneously across dimension, normalization, depth, dropout, and learning rate. |
| **RISK-05** | Statistical | **MEDIUM** | Overstatement of Paired Win Rates (Claimed 88% vs Actual 73.33%) | Paired seed-level win rate of Margin+Cosine over LC+FD was overstated in executive summaries. |
| **RISK-06** | Metric Accounting | **MEDIUM** | Conflation of Single-Seed and Multi-Seed Mean Threshold Crossings | Early reports cited Seed 42 milestones without explicitly labeling them as single-seed instances. |
| **RISK-07** | Literature | **MEDIUM** | Incommensurable Published Literature Comparisons | Comparing low-budget frozen probes against 10%–30% end-to-end trained models (HybridSN, SpectralFormer) without identical protocols. |
| **RISK-08** | Reproducibility | **LOW** | Minor Floating-Point Std Deviation Differences (ddof=1) | Discrepancy of $< 5 \times 10^{-5}$ between manual scripts due to degrees of freedom conventions. |

---

## Detailed Risk Analyses & Mitigations

### RISK-01: Spatial Autocorrelation Leakage via Random Pixel Partitioning
- **Severity:** **CRITICAL**
- **Evidence:** Source code in `src/datasets/indian_pines.py`, `salinas.py`, and `pavia_university.py` lines 52–66 extracts ground truth pixel coordinates `np.argwhere(labels > 0)`, executes a global pseudo-random shuffle `np.random.shuffle(coords)`, and slices them into train/val/test splits. Around each sampled pixel coordinate, a spatial patch of size $15 \times 15$ is extracted.
- **Impact:** Neighboring spatial pixels share almost identical spatial contexts and overlapping receptive fields. When one pixel falls in train and its spatial neighbor falls in test, representation leakage occurs. This is a well-known vulnerability in remote sensing literature that inflates absolute accuracy across all methods (both Random and Active Learning).
- **Recommended Resolution:**
  1. Clearly disclose in all publications and reports that the primary benchmark uses the standard random-pixel protocol common in initial foundation model probing benchmarks.
  2. Implement and execute a true spatially disjoint block-split benchmark (e.g., disjoint geographic sub-scenes or bounding boxes) as a separate validation track before final publication.
  3. Never claim that high accuracy under pixel-wise splits guarantees spatial generalization to unseen geographic scenes.

---

### RISK-02: Complete Absence of Conformal Guidance Implementation
- **Severity:** **CRITICAL**
- **Evidence:** Repository-wide code inspection reveals that the only calibration logic present is Temperature Scaling (Platt scaling) for logit calibration (`src/learning_engine/temperature_scaling.py`) and Expected Calibration Error (`expected_calibration_error`). There is **zero** implementation of split-conformal prediction, conformal quantile calibration, nonconformity score functions, prediction set generation, or conformal-set-size query filtering in any active learning loop.
- **Impact:** The project is entitled **ConformAL-HSI** ("Conformal-Guided Active Learning"), but the active learning strategies evaluated to date (LC+FD, Margin+Cosine, BALD, BADGE, K-Center) are purely standard or heuristic active learning methods. Calling Margin+Cosine "ConformAL-HSI" without conformal prediction is scientifically inaccurate.
- **Recommended Resolution:**
  1. Formally classify conformal guidance as a **Planned Framework Component** rather than an evaluated feature.
  2. In the paper, either:
     - Implement the true conformal acquisition module (using calibration set nonconformity quantiles to select samples with high set cardinality or uncertain coverage) and benchmark it, OR
     - Rebrand the acquisition study as an Active Learning Foundation Benchmark and reserve conformal guidance for the complete system release.

---

### RISK-03: Discrepancy Between Claimed (10/10/80) and Actual (70/15/15) Split
- **Severity:** **HIGH**
- **Evidence:** `results/final_research_validation/final_research_audit.md` claimed: "Indian Pines: 10% Train (1,024 pixels), 10% Val (1,025 pixels), 80% Test (8,200 pixels)." However, loading `cache/preextracted_features/indian_pines_train_feats.pt` and `_test_feats.pt` proves the tensors contain 7,174 train samples (70.00%), 1,537 val samples (15.00%), and 1,538 test samples (15.01%).
- **Impact:** While all methods were evaluated on the exact same cached tensors (ensuring perfect internal fairness), the documented description of the test set size was factually erroneous by a factor of 5.3x.
- **Recommended Resolution:**
  1. Correct all documentation to state the verified split: 70% candidate training pool, 15% validation pool, and 15% strictly held-out test pool.
  2. Emphasize that active learning candidates were drawn strictly from the 70% training pool, and the 15% test pool was never accessed during training or acquisition.

---

### RISK-04: Confounded Head A vs Head B Capacity Comparisons
- **Severity:** **HIGH**
- **Evidence:** In `run_classifier_capacity_interaction_study.py`, Head A was defined without any normalization layer (`Linear(2048, 256) -> ReLU -> Linear(256, C)`) and trained with `lr=2e-3` for 45 epochs, leading to training divergence and a collapse to ~32% OA on Salinas. Meanwhile, Head B featured 2 hidden layers (512 and 256), BatchNorm1d, Dropout(0.1), and input L2-normalization.
- **Impact:** Attributing the performance difference between Head A and Head B purely to "parameter capacity" is scientifically invalid because hidden dimension, network depth, normalization type, regularization, and optimization stability were altered simultaneously.
- **Recommended Resolution:**
  1. Qualify all capacity claims: Head B's superiority stems from the combination of **representation capacity, normalization stability (BatchNorm), and regularization (Dropout)**, rather than parameter count alone.
  2. Clearly document that Head A in the three-way comparison uses `ClassifierHead` with `LayerNorm(2048)` and `LayerNorm(256)` trained at `lr=5e-4` for 15 epochs, achieving 61.34% OA on Salinas, not 32%.

---

### RISK-05: Overstatement of Paired Win Rates
- **Severity:** **MEDIUM**
- **Evidence:** The three-way comparison report claimed Margin+Cosine beats LC+FD in "88% of paired round-by-round evaluations across all five seeds." Independent recomputation across all 150 post-initial pairs (rounds 1–10 across 3 datasets and 5 seeds) reveals the actual win rate is **73.33% (110 wins, 0 ties, 40 losses)**.
- **Impact:** The claim was mathematically overstated by 14.7%. While 73.33% remains highly statistically significant ($p = 1.03 \times 10^{-9}$), reporting incorrect percentages undermines scientific credibility.
- **Recommended Resolution:**
  1. Update all reports and manuscripts with the exact empirical win rate: **73.33% across all post-initial rounds** (78.0% on Indian Pines, 76.0% on Salinas, 66.0% on Pavia University).
  2. Include full paired statistical test values ($t = 6.52, p < 10^{-8}$) to provide rigorous statistical backing.

---

### RISK-06: Conflation of Single-Seed and Multi-Seed Mean Threshold Crossings
- **Severity:** **MEDIUM**
- **Evidence:** Single-seed verification on Seed 42 showed Margin+Cosine crossing 60% OA at N=360 (IP), N=240 (Salinas), and N=250 (Pavia). When 5 seeds were aggregated, the ensemble mean crossed 60% OA at >360 (IP), N=220 (Salinas), and N=290 (Pavia).
- **Impact:** Readers comparing early notes against the final multi-seed report suspected experimental manipulation or inconsistent results.
- **Recommended Resolution:**
  1. Formally publish `seed_level_threshold_analysis.csv` explicitly delineating individual seed trajectories from 5-seed mean curves.
  2. Never present single-seed milestone numbers without explicitly stating "(Seed 42 only)".

---

### RISK-07: Incommensurable Published Literature Comparisons
- **Severity:** **MEDIUM**
- **Evidence:** Literature models such as HybridSN (Roy et al., 2020) and SpectralFormer (Hong et al., 2021) report 98%–99% OA using 10% to 30% of total labeled samples (e.g., 3,074 labels on Indian Pines), trained end-to-end with deep spatial-spectral 3D convolutions.
- **Impact:** Claiming that ConformAL-HSI beats 98% literature models at low budgets ($N \in [100, 260]$) is invalid because published papers did not report or evaluate low-budget checkpoints ($N < 300$).
- **Recommended Resolution:**
  1. Clearly restrict literature claims to what was actually verified: ConformAL-HSI operates at low budgets ($N \in [100, 360]$) where full end-to-end retraining from scratch is computationally infeasible and prone to catastrophic small-sample overfitting.
  2. Concede that at massive sample budgets ($N \ge 2,000$), our frozen probe reaches 92.69% on IP, 89.20% on Salinas, and 86.76% on Pavia, while end-to-end models reach ~98% because their backbone representations adapt to domain-specific sensor bands.

---

### RISK-08: Minor Floating-Point Std Deviation Differences
- **Severity:** **LOW**
- **Evidence:** Discrepancy of $< 5 \times 10^{-5}$ observed between standard deviation calculations due to default degrees of freedom (`ddof=0` vs `ddof=1`).
- **Impact:** Negligible numerical variation.
- **Recommended Resolution:**
  Standardize all standard deviation reporting to sample standard deviation (`ddof=1`).

