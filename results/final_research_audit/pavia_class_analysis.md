# Pavia University: Forensic Class-Wise Dynamics & Spatial Autocorrelation Analysis

## 1. Executive Summary & Verification Mandate

This report provides the audited forensic analysis of why passive Random Sampling consistently outperforms Active Learning on the high-resolution Pavia University benchmark ($80.66\% \pm 0.47\%$ OA vs. $73.46\% \pm 2.09\%$ OA at $N = 980$ in Exp 1, and $80.57\% \pm 0.11\%$ vs. $75.35\% \pm 2.56\%$ in Track B, reporting sample standard deviation with $\text{ddof}=1$).

We investigate the exact class-level metrics, resolve earlier single-seed script artifacts, and test the spatial boundary concentration hypothesis directly against the dataset geometry.

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

## 3. Audited Class-Wise Allocation and Recall Breakdown (N=980)

To rigorously analyze how each strategy allocates its 980 training labels, we evaluate allocations across seeds `42, 43, 44` (each starting from exactly 5 initial seed labels per class = 45 initial labels):

| Class ID | Land-Cover Class Name | Test Support | Test Share (%) | Random Labels ($N=980$) | Random Share (%) | AL Labels ($N=980$) | AL Share (%) | Allocation Delta (AL $-$ Rand) | Random Recall (%) | AL Recall (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | Asphalt | 960 | 14.96% | $150.0 \pm 12.0$ | 15.31% | $276.0 \pm 84.3$ | 28.16% | $+126.0\text{ (+12.9%)}$ | $74.16\%$ | $70.73\%$ |
| **1** | **Meadows** | **2,763** | **43.06%** | $\mathbf{417.3 \pm 7.0}$ | **42.59%** | $\mathbf{223.3 \pm 96.1}$ | **22.79%** | $\mathbf{-194.0\text{ (-19.8%)}}$ | $\mathbf{90.40\%}$ | $\mathbf{80.55\%}$ |
| **2** | Gravel | 296 | 4.61% | $52.3 \pm 10.1$ | 5.34% | $37.3 \pm 18.7$ | 3.81% | $-15.0\text{ (-1.5%)}$ | $61.37\%$ | $66.78\%$ |
| **3** | Trees | 517 | 8.06% | $68.0 \pm 8.9$ | 6.94% | $71.0 \pm 6.9$ | 7.25% | $+3.0\text{ (+0.3%)}$ | $75.76\%$ | $72.53\%$ |
| **4** | Painted Metal | 201 | 3.13% | $37.0 \pm 6.0$ | 3.78% | $9.7 \pm 3.1$ | 0.99% | $-27.3\text{ (-2.8%)}$ | $98.18\%$ | $93.37\%$ |
| **5** | Bare Soil | 774 | 12.06% | $119.3 \pm 9.5$ | 12.17% | $156.7 \pm 54.5$ | 15.99% | $+37.3\text{ (+3.8%)}$ | $72.95\%$ | $70.54\%$ |
| **6** | Bitumen | 193 | 3.01% | $27.7 \pm 3.1$ | 2.82% | $48.3 \pm 19.5$ | 4.93% | $+20.7\text{ (+2.1%)}$ | $53.02\%$ | $52.33\%$ |
| **7** | Self-Blocking Bricks| 565 | 8.80% | $81.0 \pm 8.5$ | 8.26% | $103.7 \pm 50.3$ | 10.58% | $+22.7\text{ (+2.3%)}$ | $71.56\%$ | $71.62\%$ |
| **8** | Shadows | 148 | 2.31% | $27.3 \pm 4.0$ | 2.79% | $54.0 \pm 17.5$ | 5.51% | $+26.7\text{ (+2.7%)}$ | $79.96\%$ | $79.95\%$ |
| **Total / OA** | — | **6,417** | **100%** | **980** | **100%** | **980** | **100%** | — | $\mathbf{80.57\%}$ | $\mathbf{75.35\%}$ |

*Note*: Initial seed is exactly 5 labels per class across all 9 classes (45 initial labels). An earlier scratch script (`audit_pavia_class_metrics.py`) simulated an artificial one-shot batch of 935 samples on Seed 42, which erroneously recorded 494 Meadow labels; the true multi-round active learning benchmark allocates an average of $223.3 \pm 96.1$ labels (22.79%) to Meadows.

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

### The Primary Statistically Supported Cause: Dominant-Class Underrepresentation
The primary reason Random outperforms Active Learning on Pavia University is **Dominant-Class Underrepresentation**:
1. Active learning diverts queries into high-uncertainty urban classes (over-allocating Asphalt by $+12.9\%$ and Bare Soil by $+3.8\%$) while allocating **only $22.79\%$ of its budget to Meadows**, nearly half of Meadows' actual $43.84\%$ area presence.
2. Because Meadows constitutes **$43.06\%$ of all test pixels**, under-allocating training samples to Meadows reduces test recall on Meadows from **$90.40\%$ down to $80.55\%$**.
3. Because Overall Accuracy is an area-weighted metric, this $9.85\%$ recall loss on the $43.06\%$ test sector directly pulls scene-wide Overall Accuracy down by $\approx 4.24\%$, accounting for the observed deficit ($75.35\%$ vs. $80.57\%$).
4. Meanwhile, uniform random sampling naturally preserves the geographic area prior ($42.59\%$ to Meadows), providing balanced coverage across dominant and minority classes.
