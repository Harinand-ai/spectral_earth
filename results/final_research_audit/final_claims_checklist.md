# ConformAL-HSI: Final Research Claims Checklist & Paper Writing Guardrails

**Date:** November 2026  
**Document Type:** Scientific Audit, Claim Verification & Integrity Checklist  
**Status:** Approved for Paper Writing  

---

## 1. Verified Core Claims (Safe to Use in Paper)

| Claim | Empirical Evidence | Status |
|---|---|:---:|
| **1. Label Efficiency on Indian Pines** | Proposed Cosine AL reaches 75% OA at $N=645$ labels ($75.96\%$), whereas Random Sampling requires $N=980$ ($83.03\%$), yielding **$+34.18\%$ measured label savings** (335 fewer manual labels). | **VERIFIED** |
| **2. Class-Balanced Average Accuracy Gains** | On Indian Pines ($N=980$), Proposed AL achieves **$84.26\%$ AA** vs. $78.61\%$ for Random (**$+5.65\%$**), and **$0.8402$ Macro-F1** vs. $0.7843$ (**$+0.0559$**). Minority classes receive significant recall boosts (Class 3 Corn $+18.1\%$, Class 4 Grass $+10.8\%$, Class 8 Oats $+16.7\%$, Class 11 Soybean $+10.7\%$, Class 14 Buildings $+21.1\%$). | **VERIFIED** |
| **3. Foundation Embedding Representation Superiority** | A trained bottleneck MLP head on frozen SpectralEarth 3D-ResNet50 embeddings ($\nabla_\theta = 0$) achieves **$84.20\%–84.64\%$ OA** across datasets, dramatically outperforming a linear probe ($28.2\%–41.2\%$) and SVM ($65.2\%–67.2\%$), and beating Random Forest ($74.0\%$ on Indian Pines). | **VERIFIED** |
| **4. Dataset-Dependent Behavior on Salinas** | Salinas exhibits high spatial continuity and low class imbalance; Random Sampling and Proposed AL perform at near-parity at moderate budgets ($84.64\%$ vs $81.56\%$ at $N=980$, overlapping within error bars at $N=100$: $66.82\%$ vs $64.70\%$). | **VERIFIED** |
| **5. Random Sampling Advantage on Pavia University** | On Pavia University, Random Sampling outperforms Active Learning in pixel Overall Accuracy across all budgets ($80.57\%$ vs $75.35\%$ at $N=980$). This is driven by **Class Allocation Mismatch**: Meadows occupies $43.06\%$ of the test set; Random naturally allocates $42.59\%$ of its training budget to Meadows, whereas AL diverts labels toward ambiguous boundary classes (Asphalt, Soil, Bricks), allocating only $22.79\%$ to Meadows. | **VERIFIED** |
| **6. Computational Efficiency of Lightweight MLP Head** | Inference on 28,488 test samples (Salinas) takes **$0.035\text{ seconds}$** ($1.62\times 10^6$ samples/s) on an RTX 2050 GPU, compared to **$33.76\text{ seconds}$** for CPU scikit-learn SVM (RBF kernel), and trains in **$3.1\text{ seconds}$** for 45 epochs. | **VERIFIED** |

---

## 2. Refined & Corrected Findings

| Original Unrefined Statement | Corrected Scientific Statement | Reason for Correction |
|---|---|---|
| *"Active Learning starves Meadows on Seed 42 with 494 labels"* | Active Learning allocates an average of **$223.3 \pm 96.1$ labels ($22.79\%$)** to Meadows across seeds 42–44, compared to **$417.3 \pm 7.0$ labels ($42.59\%$)** for Random Sampling. | The "494 labels" figure was an artifact of an artificial single-stage query script (`audit_pavia_class_metrics.py`). In the true iterative pipeline, AL consistently under-allocates to Meadows by $-19.8\%$ relative to the true area prior. |
| *"Active learning requires fewer labels at all accuracy thresholds"* | Active learning requires fewer labels at **moderate-to-high accuracy thresholds** on imbalanced datasets (e.g. 75% OA on Indian Pines: $+34.18\%$ savings). At low thresholds ($60\%$ OA), passive random sampling crosses earlier ($180$ vs $220$ labels, **$-22.22\%$ relative savings**). | Discrete experimental records show that passive random sampling quickly learns coarse representations for dominant classes, while AL focuses on boundary/uncertainty refinement. |
| *"Pavia performance deficit is caused by spatial boundary concentration"* | Direct spatial distance transform on `PaviaU_gt` shows the mean distance to boundaries is $7.48 \pm 0.16\text{ px}$ for Random vs $6.42 \pm 2.79\text{ px}$ for AL ($p=0.570$, not significant). The true cause is **Class Allocation Mismatch** relative to the $43.8\%$ geographic prior. | Spatial hypothesis tested and disproven; class allocation distribution confirmed as primary mechanism. |
| *"ConformAL-HSI MLP Head is 965x faster than SVM"* | ConformAL-HSI (PyTorch, RTX 2050 GPU) processes $1.62\times 10^6\text{ samples/s}$ vs. SVM (scikit-learn LIBSVM, single-threaded CPU) at $1,687\text{ samples/s}$. | The throughput difference reflects both algorithmic complexity ($O(N \cdot d)$ vs $O(N \cdot M_{\text{supp}} \cdot d)$) AND hardware platform (GPU vs CPU). Hardware conditions must always be disclosed. |

---

## 3. Unsupported Claims (Strictly Prohibited in Research Paper)

1. **DO NOT claim that HybridSN, SpectralFormer, or SSFTT were experimentally benchmarked**:
   - They were investigated via computational feasibility analysis and referenced against published literature protocols.
   - Any discussion of them must be clearly labeled as **Literature Reference Comparison**, citing the original publications (Roy et al., 2020; Hong et al., 2021; Sun et al., 2022).
2. **DO NOT claim that Active Learning is universally superior across all HSI datasets**:
   - The paper must honestly present the Category A (Indian Pines: AL advantage), Category B (Salinas: near-parity), and Category C (Pavia: Random advantage) taxonomy.
   - Presenting Pavia's results honestly strengthens the paper by providing mechanistic insights into how spatial land-cover area distribution interacts with active acquisition functions.
3. **DO NOT claim that end-to-end deep models "catastrophically collapse" at low budgets without empirical verification**:
   - Frame the comparison around model capacity and inductive bias: end-to-end models with millions of parameters require 10%–20% of the dataset to optimize spatial-spectral filters from scratch, whereas frozen foundation models leverage pre-trained representations that generalize with $\le 1\%$ labeled data.
4. **DO NOT report interpolated threshold estimates as measured checkpoints**:
   - Only cite discrete experimental checkpoints ($N \in \{100, 140, 180, 220, 260, 645, 980\}$ in Exp 1; $N \in \{100, 260, 980\}$ in Track B).

---

## 4. Discrepancy Reconciliation Summary (Exp 1 vs. Track B)

All differences between Experiment 1 and Track B at $N=980$ are fully explained by **Acquisition Step Granularity** and **PRNG draw sequence**:
- **Experiment 1 (7-stage active learning schedule)**: Updates intermediate models at budgets $80, 100, 140, 180, 220, 260, 645$, refining decision boundaries before each small query batch.
- **Track B (3-stage external benchmark schedule)**: Evaluates non-deep baselines at budgets $100, 260, 980$, selecting a larger batch of 720 samples from the $N=260$ model.
- Both schedules are methodologically valid and confirm identical relative performance hierarchies.

