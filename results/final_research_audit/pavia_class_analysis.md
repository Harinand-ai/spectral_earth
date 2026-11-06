# Pavia University: Forensic Class-Wise Dynamics & Spatial Autocorrelation Analysis

## 1. Executive Summary & Verification Mandate

This report provides the audited forensic analysis of why passive Random Sampling consistently outperforms Active Learning on the high-resolution Pavia University benchmark ($80.66\% \pm 0.47\%$ OA vs. $73.46\% \pm 2.09\%$ OA at $N = 980$, and $66.79\% \pm 1.12\%$ vs. $56.05\% \pm 2.37\%$ at $N = 260$, reporting sample standard deviation with $\text{ddof}=1$).

We investigate the exact class-level metrics and test the spatial boundary concentration hypothesis directly against the dataset geometry.

---

## 2. Dataset Geographic Composition

Pavia University features an extreme land-cover area distribution:

| Class ID | Land-Cover Class Name | Train Pool Count | Train Pool Share (%) | Test Set Count | Test Set Share (%) |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Asphalt | 4,629 | 15.46% | 960 | 14.96% |
| **1** | **Meadows** | **13,127** | **43.84%** | **2,763** | **43.06%** |
| **2** | Gravel | 1,472 | 4.92% | 296 | 4.61% |
| **3** | Trees | 2,078 | 6.94% | 517 | 8.06% |
| **4** | Painted Metal Sheets | 951 | 3.18% | 201 | 3.13% |
| **5** | Bare Soil | 3,524 | 11.77% | 774 | 12.06% |
| **6** | Bitumen | 937 | 3.13% | 193 | 3.01% |
| **7** | Self-Blocking Bricks | 2,557 | 8.54% | 565 | 8.80% |
| **8** | Shadows | 668 | 2.23% | 148 | 2.31% |
| **Total** | — | **29,943** | **100.00%** | **6,417** | **100.00%** |

**Key Property**: Class 1 (*Meadows*) occupies **43.84%** of the entire training pool and **43.06%** of the test set. Together, Asphalt ($15.5\%$), Meadows ($43.8\%$), and Bare Soil ($11.8\%$) account for **over 71%** of the scene.

---

## 3. Audited Class-Wise Breakdown at Budget 980 (Seed 42)

To rigorously analyze how each strategy allocated its 980 training labels, we distinguish **Initial Seed Labels** (5 per class = 45 total) from **Newly Acquired Active Labels**, reporting total support, test recall, precision, and F1 scores:

| Class Name | Test Support | Random (Init + Acq = Total) | AL (Init + Acq = Total) | Random Recall | AL Recall | Random Precision | AL Precision | Random F1 | AL F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Asphalt** | 960 | 5 + 157 = **162** | 5 + 120 = **125** | **71.04%** | 45.00% | 72.17% | 66.87% | **0.7160** | 0.5383 |
| **Meadows** | 2,763 | 5 + 405 = **410** | 5 + 489 = **494** | **90.12%** | 86.93% | 87.77% | 89.78% | **0.8893** | 0.8834 |
| **Gravel** | 296 | 5 + 46 = **51** | 5 + 0 = **5** | **69.26%** | 67.57% | 59.08% | 34.84% | **0.6376** | 0.4603 |
| **Trees** | 517 | 5 + 73 = **78** | 5 + 81 = **86** | 76.60% | **82.79%** | 80.49% | 81.37% | 0.7849 | **0.8207** |
| **Painted Metal** | 201 | 5 + 32 = **37** | 5 + 0 = **5** | **99.50%** | 90.05% | 94.79% | 71.04% | **0.9709** | 0.7939 |
| **Bare Soil** | 774 | 5 + 105 = **110** | 5 + 192 = **197** | **69.51%** | 68.99% | 75.35% | 67.42% | **0.7231** | 0.6820 |
| **Bitumen** | 193 | 5 + 26 = **31** | 5 + 0 = **5** | **55.44%** | 48.19% | 66.46% | 37.35% | **0.6045** | 0.4208 |
| **Self-Blocking Bricks**| 565 | 5 + 68 = **73** | 5 + 3 = **8** | **67.96%** | 45.49% | 71.64% | 53.43% | **0.6975** | 0.4914 |
| **Shadows** | 148 | 5 + 23 = **28** | 5 + 50 = **55** | 83.11% | **87.84%** | 70.87% | 58.82% | **0.7640** | 0.7046 |
| **Overall Accuracy (OA)** | — | — | — | **79.87%** | **72.57%** | — | — | — | — |
| **Average Accuracy (AA)** | — | — | — | **75.84%** | **69.20%** | — | — | — | — |
| **Macro F1 Score** | — | — | — | — | — | **0.7540** | **0.6277** | **0.7542** | **0.6439** |

*Note*: Initial seed is exactly 5 labels per class across all 9 classes ($45$ initial labels).

---

## 4. Empirical Evaluation of the Spatial Sampling Hypothesis

We evaluated the hypothesis that active learning selects more boundary pixels than random sampling by computing the exact Euclidean distance from every acquired training sample coordinate $(r, c)$ to the nearest class boundary in `paviaU_gt`:

$$\text{dist}(p, \mathcal{B}) = \min_{b \in \mathcal{B}} \|p - b\|_2$$

Evaluating across seeds 42, 43, and 44 ($935$ newly acquired samples per seed):

| Metric | Random Sampling | Proposed Cosine AL | Statistical Significance ($t$-test) |
|---|:---:|:---:|:---:|
| **Mean Distance to Class Boundary (px)** | **$7.48 \pm 0.16\text{ px}$** | **$6.42 \pm 2.79\text{ px}$** | $p = 0.570$ (Not statistically significant) |
| **Boundary Concentration ($\le 1.0\text{ px}$)** | **$40.6\% \pm 0.7\%$** | **$41.7\% \pm 5.5\%$** | $p = 0.742$ (Not statistically significant) |

### Key Scientific Conclusion
The empirical evidence indicates that **the boundary concentration hypothesis is NOT the primary cause of the performance difference**. Active learning samples boundaries at virtually the same rate as random sampling ($41.7\%$ vs. $40.6\%$, difference not significant).

### The Primary Statistically Supported Cause: Class Allocation Mismatch
The primary reason Random outperforms Active Learning on Pavia University is **Class Underrepresentation / Starvation**:
1. Active learning queried **0 additional samples** for Gravel, Painted Metal, and Bitumen, and **only 3 additional samples** for Bricks (retaining only 5 to 8 total labels).
2. Consequently, recall on Self-Blocking Bricks fell to **$45.49\%$** (vs. $67.96\%$ in Random) and recall on Asphalt fell to **$45.00\%$** (vs. $71.04\%$ in Random).
3. Meanwhile, uniform random sampling naturally allocated $51$ samples to Gravel, $37$ to Metal, $31$ to Bitumen, and $73$ to Bricks, providing sufficient support across all classes while maintaining $410$ samples ($41.8\%$) in the dominant Meadow class.
