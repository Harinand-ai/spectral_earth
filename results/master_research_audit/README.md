# ConformAL-HSI: Master Research Audit & Verification Suite

This directory contains the complete artifacts, reconciled metrics, statistical analyses, and methodological risk assessments resulting from the forensic research audit of **ConformAL-HSI**.

---

## Directory Contents

| File | Description |
|:---|:---|
| [`master_audit_report.md`](master_audit_report.md) | Comprehensive forensic research audit report addressing all 18 prompt areas. |
| [`experiment_reconciliation.csv`](experiment_reconciliation.csv) | Chronological classification and reconciliation of all historical experiments. |
| [`corrected_three_way_results.csv`](corrected_three_way_results.csv) | Verified aggregated metrics across 5 seeds with sample standard deviations (`ddof=1`). |
| [`seed_level_threshold_analysis.csv`](seed_level_threshold_analysis.csv) | Full seed-by-seed breakdown of 60% OA threshold crossings vs 5-seed mean curves. |
| [`paired_statistical_analysis.csv`](paired_statistical_analysis.csv) | Paired $t$-test and Wilcoxon signed-rank test results for Margin+Cosine vs LC+FD and Random. |
| [`methodological_risk_register.md`](methodological_risk_register.md) | Ranked register of methodological risks (Critical, High, Medium, Low) with mitigation plans. |
| [`publication_claims.md`](publication_claims.md) | Strict four-tier scientific claims taxonomy (Supported, Qualified, Hypotheses, Insupportable). |
| [`README.md`](README.md) | Reproduction guidelines and executive orientation. |

---

## How to Reproduce the Audited Results

### 1. Environment & Prerequisites
Ensure the environment is activated and dependencies are installed:
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Required packages: `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.

### 2. Verify Pre-Extracted Features
All experiments operate on frozen 2048-dimensional features extracted by the SpectralEarth 3D-ResNet50 backbone:
```bash
python -c "import torch; print(torch.load('cache/preextracted_features/indian_pines_train_feats.pt', weights_only=False)['feats'].shape)"
```
Expected output: `torch.Size([7174, 2048])`.

### 3. Reproduce Controlled Three-Way Active Learning Comparison
To re-run the 495 checkpoint evaluations (3 datasets $\times$ 3 strategies $\times$ 5 seeds $\times$ 11 rounds):
```bash
python run_three_way_al_comparison.py
```
This writes raw records to `results/original_vs_updated_active_learning/three_way_al_seed_results.csv`.

### 4. Re-generate Audit Tables & Statistical Analyses
To recompute all audit metrics, paired statistical tests, threshold crossings, and reconciliation tables:
```bash
python scratch/generate_audit_deliverables.py
```
All recomputed CSV files will be refreshed directly in `results/master_research_audit/`.
