# ConformAL-HSI: Extended Label-Budget Accuracy Report (N=220 to 2000)

## 1. Executive Summary and Primary Research Findings
This study extended the ConformAL-HSI framework across higher labeled budgets ($N = 220, 300, 400, 500, 700, 1000, 1500, 2000$) to address the central research question:

> **Can increasing the number of labeled pixels allow the best existing active learning methods to achieve 90%, 95%, or even 99% test accuracy, and how does Random Sampling perform when given identical budgets?**

### Key Quantitative Findings:
### Indian Pines
- **Selected AL Strategy:** Margin + Cosine Diversity
- **Peak AL OA:** **92.69% ± 0.61%** at Budget $N=2000$
- **Peak Random OA:** **88.04% ± 0.35%** at Budget $N=2000$
- **Net Difference:** +4.66% OA

### Salinas
- **Selected AL Strategy:** Margin + Cosine Diversity
- **Peak AL OA:** **89.20% ± 0.33%** at Budget $N=2000$
- **Peak Random OA:** **87.56% ± 0.47%** at Budget $N=2000$
- **Net Difference:** +1.64% OA

### Pavia University
- **Selected AL Strategy:** Margin Sampling
- **Peak AL OA:** **86.76% ± 0.96%** at Budget $N=2000$
- **Peak Random OA:** **84.52% ± 0.20%** at Budget $N=2000$
- **Net Difference:** +2.24% OA


---

## 2. Answers to the Core Research Questions

### 1. Does increasing the number of labeled pixels consistently improve accuracy?
**Yes, significantly across all three datasets.**  
- **Indian Pines:** OA increases from $61.78\% \pm 1.53\%$ ($N=220$) to **$92.69\% \pm 0.61\%$** ($N=2000$) under Active Learning (+30.91% OA), and from $62.55\%$ to $88.04\%$ under Random Sampling (+25.49% OA).
- **Salinas:** OA increases from $75.78\% \pm 1.16\%$ ($N=220$) to **$89.20\% \pm 0.33\%$** ($N=2000$) under Active Learning (+13.42% OA), and from $73.87\%$ to $87.56\%$ under Random (+13.69% OA).
- **Pavia University:** OA increases from $69.33\% \pm 1.52\%$ ($N=220$) to **$86.76\% \pm 0.96\%$** ($N=2000$) under Active Learning (+17.43% OA), and from $67.75\%$ to $84.52\%$ under Random (+16.77% OA).

### 2. What is the highest observed OA for each dataset?
- **Indian Pines:** Reaches **92.69% ± 0.61% OA** (Margin + Cosine Diversity at $N=2000$).
- **Salinas:** Reaches **89.20% ± 0.33% OA** (Margin + Cosine Diversity at $N=2000$).
- **Pavia University:** Reaches **86.76% ± 0.96% OA** (Margin Sampling at $N=2000$).

### 3. Does any dataset reach 90%, 95%, or 99%?
- **90% Threshold:** **YES, Indian Pines successfully crosses 90% OA**, reaching **90.64% ± 1.14% OA at $N=1500$** and **92.69% ± 0.61% OA at $N=2000$**. In contrast, Random Sampling fails to reach 90% even with 2000 labels ($88.04\% \pm 0.35\%$). Salinas approaches 90% at $89.20\%$ ($N=2000$), while Pavia University reaches $86.76\%$.
- **95% Threshold:** **Not reached by any dataset** (Highest: Indian Pines at 92.69% at $N=2000$).
- **99% Threshold:** **Not reached by any dataset.**
- *Scientific Honesty Assessment:* While deep spatial networks trained on overlapping random patches can report >95% due to spatial autocorrelation leakage, linear/MLP heads trained on frozen SpectralEarth foundation embeddings reach empirical representation ceilings between 86% and 93% on strictly held-out test sets.

### 4. At what budget does each method begin to saturate?
- **Saturation Knee Point:** Saturation begins between **$N = 700$ and $N = 1000$**.
- On Indian Pines, expanding from $N=220$ to $N=700$ yields $+22.42\%$ OA ($61.78\% 	o 84.20\%$), whereas tripling the budget from $N=700$ to $N=2000$ yields $+8.49\%$ OA ($84.20\% 	o 92.69\%$).
- On Salinas, expanding from $N=220$ to $N=700$ yields $+9.59\%$ OA ($75.78\% 	o 85.37\%$), while expanding from $N=700$ to $N=2000$ yields only $+3.83\%$ OA.

### 5. At identical label budgets, does active learning outperform random sampling?
- **YES, across all datasets and across almost all budgets.**
- **Indian Pines:** Active Learning consistently beats Random Sampling: $+4.30\%$ at $N=400$ ($74.81\%$ vs $70.51\%$), $+4.95\%$ at $N=500$ ($79.56\%$ vs $74.60\%$), $+5.67\%$ at $N=700$ ($84.20\%$ vs $78.53\%$), $+5.67\%$ at $N=1000$ ($88.11\%$ vs $82.44\%$), $+4.25\%$ at $N=1500$ ($90.64\%$ vs $86.39\%$), and $+4.66\%$ at $N=2000$ ($92.69\%$ vs $88.04\%$).
- **Salinas:** Active Learning maintains a consistent $+1.6\%$ to $+3.5\%$ OA lead across all budgets ($+3.51\%$ at $N=400$, $+2.36\%$ at $N=1000$, $+1.64\%$ at $N=2000$).
- **Pavia University:** Active Learning maintains a $+0.8\%$ to $+2.2\%$ OA lead across all budgets ($+1.56\%$ at $N=1000$, $+2.24\%$ at $N=2000$).

### 6. Does random sampling eventually catch up with active learning?
- On **Salinas**, the gap narrows from $+3.51\%$ at $N=400$ down to $+1.64\%$ at $N=2000$ as random sampling covers the dominant agricultural fields.
- However, on **Indian Pines and Pavia University, Random Sampling never catches up with Active Learning**. On Indian Pines, Active Learning maintains a massive $+4.66\%$ advantage at $N=2000$ ($92.69\%$ vs $88.04\%$), because rare tail classes (e.g. grass-pasture-mowed, oats, alfalfa) remain severely under-sampled under random selection.

### 7. Which dataset benefits most from additional labeled pixels?
- **Indian Pines benefits the most**, gaining **+30.91% OA** from $N=220$ to $N=2000$ under Active Learning. The high spectral similarity among 16 crop/vegetation classes requires substantial boundary-informative samples to establish clear decision hyperplanes.

### 8. Does active learning maintain its advantage at larger budgets?
- **YES.** Active Learning delivers significant discrete label savings at high accuracy thresholds:
  - **Indian Pines at 85% OA:** AL reaches the threshold at $N=1000$ ($88.11\%$), whereas Random requires $N=1500$ ($86.39\%$) -> **33.33% discrete savings (500 labels saved)**.
  - **Indian Pines at 90% OA:** AL reaches the threshold at $N=1500$ ($90.64\%$), while Random fails to reach 90% even at $N=2000$ ($88.04\%$) -> **AL Solely Achieved (>500 labels saved)**.
  - **Salinas at 85% OA:** AL reaches the threshold at $N=700$ ($85.37\%$), whereas Random requires $N=1500$ ($86.29\%$) -> **53.33% discrete savings (800 labels saved)**.
  - **Pavia University at 85% OA:** AL reaches the threshold at $N=2000$ ($86.76\%$), whereas Random fails to cross 85% ($84.52\%$) -> **AL Solely Achieved**.

---

## 3. Measured Threshold Crossings and Discrete Label Savings

The table below reports measured threshold crossings from `extended_budget_label_savings.csv`:

| Dataset | Target Threshold | Active Learning Budget | Actual AL OA | Random Budget | Actual Random OA | Discrete Savings (%) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Indian Pines | 60% OA | 220 | 61.78% | 220 | 62.55% | 0.0% | Both Reached |
| Indian Pines | 65% OA | 300 | 69.79% | 300 | 66.76% | 0.0% | Both Reached |
| Indian Pines | 70% OA | 400 | 74.81% | 400 | 70.51% | 0.0% | Both Reached |
| Indian Pines | 75% OA | 500 | 79.56% | 700 | 78.53% | 28.57% | Both Reached |
| Indian Pines | 80% OA | 700 | 84.20% | 1000 | 82.44% | 30.0% | Both Reached |
| Indian Pines | 85% OA | 1000 | 88.11% | 1500 | 86.39% | 33.33% | Both Reached |
| Indian Pines | 90% OA | 1500 | 90.64% | >2000 | 88.04% | > 0.0% | AL Solely Achieved |
| Salinas | 60% OA | 220 | 75.78% | 220 | 73.87% | 0.0% | Both Reached |
| Salinas | 65% OA | 220 | 75.78% | 220 | 73.87% | 0.0% | Both Reached |
| Salinas | 70% OA | 220 | 75.78% | 220 | 73.87% | 0.0% | Both Reached |
| Salinas | 75% OA | 220 | 75.78% | 300 | 76.67% | 26.67% | Both Reached |
| Salinas | 80% OA | 400 | 81.98% | 500 | 80.25% | 20.0% | Both Reached |
| Salinas | 85% OA | 700 | 85.37% | 1500 | 86.29% | 53.33% | Both Reached |
| Pavia University | 60% OA | 220 | 69.33% | 220 | 67.75% | 0.0% | Both Reached |
| Pavia University | 65% OA | 220 | 69.33% | 220 | 67.75% | 0.0% | Both Reached |
| Pavia University | 70% OA | 300 | 72.91% | 300 | 71.40% | 0.0% | Both Reached |
| Pavia University | 75% OA | 500 | 76.58% | 500 | 75.78% | 0.0% | Both Reached |
| Pavia University | 80% OA | 1000 | 82.01% | 1000 | 80.46% | 0.0% | Both Reached |
| Pavia University | 85% OA | 2000 | 86.76% | >2000 | 84.52% | > 0.0% | AL Solely Achieved |

---

## 4. Methodological and Spatial Autocorrelation Considerations
- **Spatial Autocorrelation in Hyperspectral Imagery:** In standard pixel-level random train/test splits, adjacent pixels often fall into both training and testing sets, which can inflate test scores.
- **Strict Linear/MLP Evaluation:** Our framework evaluates isolated pixel embeddings without spatial post-processing (e.g., Markov Random Fields or spatial convolutions) to ensure that performance reflects the intrinsic quality of the foundation embeddings.

---

## 5. Artifact Index
- Seed Results: [`extended_budget_seed_results.csv`](extended_budget_seed_results.csv)
- Aggregated Results: [`extended_budget_aggregated_results.csv`](extended_budget_aggregated_results.csv)
- Label Savings: [`extended_budget_label_savings.csv`](extended_budget_label_savings.csv)
- Per-Class Metrics: [`extended_budget_per_class_metrics.csv`](extended_budget_per_class_metrics.csv)
- Figure 1 (AL Scaling): [`al_accuracy_vs_budget.png`](al_accuracy_vs_budget.png)
- Figure 2 (Random Scaling): [`random_accuracy_vs_budget.png`](random_accuracy_vs_budget.png)
- Figure 3 (Head-to-Head Comparison): [`al_vs_random_extended_budget.png`](al_vs_random_extended_budget.png)
- Figure 4 (Label Savings): [`threshold_label_savings.png`](threshold_label_savings.png)
- Figure 5 (Computational Scaling): [`runtime_memory_scaling.png`](runtime_memory_scaling.png)
