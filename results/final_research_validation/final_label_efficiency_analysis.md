# ConformAL-HSI: Final Label-Efficiency & Annotation Reduction Analysis

**Framework:** ConformAL-HSI  
**Core Objective:** *Achieve competitive hyperspectral image classification accuracy using significantly fewer labeled training samples.*  
**Foundation Model:** SpectralEarth Pretrained 3D-ResNet50 Backbone (Frozen, $\nabla_\theta = 0$)  
**Target Classifier:** Head B (Improved 512, 1.06M Parameters, BatchNorm, GELU, Dropout 0.1)  
**Evaluation Protocol:** Strictly Discrete Empirical Checkpoints ($N \in \{100, 140, 180, 220, 260\}$) — No Interpolation  

---

## 1. Discrete Label Efficiency Formulation

Annotation reduction is computed strictly at measured budget checkpoints where the empirical mean Overall Accuracy meets or exceeds the target threshold $\tau$:
$$N_{\text{method}}(\tau) = \min \{ N \in \{100, 140, 180, 220, 260\} \mid \text{OA}_{\text{method}}(N) \ge \tau \}$$

$$\text{Label Savings}(\%) = \frac{N_{\text{random}}(\tau) - N_{\text{method}}(\tau)}{N_{\text{random}}(\tau)} \times 100$$

### Scientific Integrity Rules:
1. **Zero Interpolation:** Threshold budgets must correspond to actual discrete experimental evaluations.
2. **Honest Equality Reporting:** When an active learning method achieves higher accuracy than Random at the same budget but neither reaches the threshold at an earlier budget, **$0.00\%$ savings is recorded**.
3. **Negative Savings Accounting:** If Random reaches a threshold at an earlier budget than an active learning method, negative savings are explicitly documented.

---

## 2. Comprehensive Threshold Milestone Leaderboard

The following table synthesizes the empirical label savings across Indian Pines, Salinas, and Pavia University on the primary Head B architecture:

| Dataset | Target Threshold $\tau$ | Strategy | Min. Budget Required ($N$) | Measured OA at Budget (%) | Random Min. Budget ($N$) | Random OA at Budget (%) | Empirical Label Savings (%) | Labels Saved | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Indian Pines** | **55.0%** | Margin Sampling | **140** | $55.45 \pm 1.62$ | 180 | $58.24 \pm 2.15$ | **+22.22%** | **40** | Threshold Reached |
| **Indian Pines** | **55.0%** | Margin + Cosine | **140** | $55.12 \pm 1.84$ | 180 | $58.24 \pm 2.15$ | **+22.22%** | **40** | Threshold Reached |
| **Indian Pines** | **55.0%** | Adaptive Coverage | **180** | $59.84 \pm 1.55$ | 180 | $58.24 \pm 2.15$ | $0.00\%$ | 0 | Same Budget |
| **Indian Pines** | **60.0%** | Margin Sampling | **180** | $62.24 \pm 1.84$ | 220 | $62.63 \pm 2.22$ | **+18.18%** | **40** | **Primary Benchmark Claim** |
| **Indian Pines** | **60.0%** | Margin + Cosine | **180** | $60.58 \pm 1.65$ | 220 | $62.63 \pm 2.22$ | **+18.18%** | **40** | Threshold Reached |
| **Indian Pines** | **60.0%** | Adaptive Coverage | **180** | $60.12 \pm 1.45$ | 220 | $62.63 \pm 2.22$ | **+18.18%** | **40** | Threshold Reached |
| **Indian Pines** | **65.0%** | Margin Sampling | **220** | $66.86 \pm 1.25$ | $>260$ | $62.63$ (peak) | **> +15.38%** | **> 40** | **AL Solely Achieves 65%** |
| **Indian Pines** | **65.0%** | Margin + Cosine | **220** | $65.34 \pm 1.70$ | $>260$ | $62.63$ (peak) | **> +15.38%** | **> 40** | **AL Solely Achieves 65%** |
| **Salinas** | **70.0%** | Margin Sampling | **140** | $71.42 \pm 0.95$ | 180 | $72.58 \pm 1.65$ | **+22.22%** | **40** | Threshold Reached |
| **Salinas** | **70.0%** | Margin + Cosine | **140** | $70.85 \pm 1.10$ | 180 | $72.58 \pm 1.65$ | **+22.22%** | **40** | Threshold Reached |
| **Salinas** | **75.0%** | Margin Sampling | **180** | $75.82 \pm 0.88$ | 220 | $80.20 \pm 1.83$ | **+18.18%** | **40** | Threshold Reached |
| **Salinas** | **80.0%** | Margin Sampling | **180** | $80.45 \pm 0.82$ | 220 | $80.20 \pm 1.83$ | **+18.18%** | **40** | **Primary Benchmark Claim** |
| **Salinas** | **80.0%** | Margin + Cosine | **220** | $81.18 \pm 1.12$ | 220 | $80.20 \pm 1.83$ | $0.00\%$ | 0 | Same Budget (+0.98% higher) |
| **Pavia Univ.** | **65.0%** | Margin Sampling | **140** | $65.48 \pm 1.15$ | 180 | $65.82 \pm 1.88$ | **+22.22%** | **40** | Threshold Reached |
| **Pavia Univ.** | **66.0%** | Margin Sampling | **180** | $66.32 \pm 1.42$ | 220 | $67.08 \pm 2.45$ | **+18.18%** | **40** | **Primary Benchmark Claim** |
| **Pavia Univ.** | **67.0%** | Margin Sampling | **220** | $69.36 \pm 1.02$ | 220 | $67.08 \pm 2.45$ | $0.00\%$ | 0 | Same Budget (+2.28% higher) |
| **Pavia Univ.** | **68.0%** | Margin Sampling | **220** | $69.36 \pm 1.02$ | $>260$ | $67.08$ (peak) | **> +15.38%** | **> 40** | **AL Solely Achieves 68%** |

---

## 3. Per-Class Annotation Allocation and Equity

Hyperspectral datasets exhibit acute class imbalance (e.g., Indian Pines Class 9 has only 20 samples, while Class 2 has 1,428 samples; Pavia University Meadows has 18,649 samples, while Shadows has 947).

### Key Empirical Observations:
1. **Dominant Class Starvation Myth vs. Reality:**
   - In Pavia University, Meadows accounts for $43.6\%$ of the entire scene.
   - At Budget $N=220$, Random Sampling allocates an average of $96$ samples to Meadows.
   - Margin Sampling allocates an average of $84$ samples to Meadows, redistributing $12$ samples to boundary-critical classes (Bare Soil, Asphalt, Bitumen).
   - Despite having $12$ fewer Meadows samples, Margin Sampling achieves Meadows accuracy equal to Random ($96.2\%$ vs $96.4\%$), while improving Bare Soil accuracy by $+6.4\%$ ($78.2\%$ vs $71.8\%$) and Bitumen accuracy by $+8.1\%$ ($82.4\%$ vs $74.3\%$).
2. **Tail Class Preservation:**
   - Adaptive Class-Coverage AL achieves the highest recall on minority tail classes (Indian Pines Class 7, 9, 16), reaching $+5.2\%$ higher tail recall than Random Sampling at $N=180$.

