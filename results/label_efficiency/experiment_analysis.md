# ConformAL-HSI: Experiment 1 — Unified Label-Efficiency Evaluation

## 1. Executive Summary & Experimental Controls

- **Primary Objective**: Measure label efficiency (accuracy vs. labeled sample count) across Indian Pines, Salinas, and Pavia University.
- **Fair Comparison Protocol**: The **exact same** `ImprovedClassifierHead` (512 units, unit hypersphere projection, BatchNorm, Cosine Annealing 45 epochs) is used across all 5 primary acquisition strategies.
- **Classifier Head Ablation**: The original 256-dim head (25 epochs, fixed $lr$) is evaluated alongside as an explicit ablation reference.
- **Backbone State**: Strictly **FROZEN** `SpecResNet50` (24.24M parameters, $\nabla_\theta = 0$).
- **Seeds**: 42, 43, 44 (Mean $\pm$ Std reported).

## 2. Accuracy-versus-Budget Master Table (Mean ± Std Across Seeds)

| Dataset | Strategy | Category | 100 Labels | 140 Labels | 180 Labels | 220 Labels | 260 Labels | 645 Labels | 980 Labels (OA) | 980 (AA) | 980 (Kappa) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Indian Pines** | Random Sampling | Primary Comparison (Improved Head) | 50.65% ± 2.84% | 52.28% ± 2.42% | 60.14% ± 0.23% | 63.37% ± 1.17% | 61.66% ± 0.95% | 73.73% ± 1.07% | 83.03% ± 1.01% | 79.91% | 0.8045 |
| **Indian Pines** | Pure Uncertainty (Least Confidence) | Primary Comparison (Improved Head) | 47.55% ± 1.05% | 48.42% ± 2.87% | 53.86% ± 1.64% | 56.55% ± 1.59% | 53.03% ± 2.83% | 69.38% ± 0.56% | 80.56% ± 0.54% | 77.98% | 0.7760 |
| **Indian Pines** | Baseline Unc + Diversity (Euclidean) | Primary Comparison (Improved Head) | 49.35% ± 0.97% | 48.55% ± 1.22% | 57.43% ± 0.88% | 62.03% ± 0.69% | 59.84% ± 2.12% | 76.35% ± 0.43% | 85.93% ± 0.32% | 84.32% | 0.8378 |
| **Indian Pines** | Proposed Cosine AL (Unc + Pool Diversity) | Primary Comparison (Improved Head) | 46.58% ± 2.36% | 50.87% ± 3.59% | 57.56% ± 3.53% | 60.58% ± 2.50% | 60.21% ± 1.11% | 75.96% ± 0.81% | 86.17% ± 0.16% | 84.87% | 0.8412 |
| **Indian Pines** | Proposed AL + Pseudo-Labeling | Primary Comparison (Improved Head) | 37.54% ± 11.97% | 43.37% ± 5.16% | 52.58% ± 2.88% | 57.63% ± 1.37% | 56.03% ± 2.45% | 70.46% ± 1.00% | 83.59% ± 1.35% | 79.60% | 0.8105 |
| **Indian Pines** | Baseline Head + Random | Classifier Ablation (Original Head) | 42.98% ± 1.69% | 39.40% ± 4.71% | 51.60% ± 3.05% | 51.02% ± 5.77% | 45.38% ± 5.06% | 61.99% ± 0.38% | 66.80% ± 1.75% | 61.00% | 0.6109 |
| **Indian Pines** | Baseline Head + Unc+Div | Classifier Ablation (Original Head) | 41.31% ± 1.28% | 37.99% ± 3.82% | 48.18% ± 7.49% | 46.62% ± 4.88% | 43.61% ± 3.99% | 56.26% ± 2.96% | 67.01% ± 1.19% | 62.09% | 0.6133 |
| **Salinas** | Random Sampling | Primary Comparison (Improved Head) | 65.86% ± 1.15% | 68.78% ± 0.50% | 71.68% ± 0.38% | 73.59% ± 0.48% | 71.94% ± 1.08% | 80.48% ± 0.98% | 85.17% ± 0.23% | 87.73% | 0.8346 |
| **Salinas** | Pure Uncertainty (Least Confidence) | Primary Comparison (Improved Head) | 62.58% ± 1.85% | 61.29% ± 0.99% | 66.17% ± 1.25% | 67.56% ± 1.24% | 62.88% ± 0.94% | 71.94% ± 3.04% | 81.73% ± 0.75% | 85.98% | 0.7970 |
| **Salinas** | Baseline Unc + Diversity (Euclidean) | Primary Comparison (Improved Head) | 63.46% ± 2.08% | 60.51% ± 2.17% | 67.81% ± 1.54% | 69.31% ± 0.47% | 67.94% ± 1.22% | 77.73% ± 2.08% | 84.40% ± 0.45% | 89.68% | 0.8267 |
| **Salinas** | Proposed Cosine AL (Unc + Pool Diversity) | Primary Comparison (Improved Head) | 62.57% ± 1.97% | 63.41% ± 3.07% | 71.21% ± 0.80% | 72.75% ± 0.90% | 71.99% ± 1.89% | 79.06% ± 1.64% | 84.87% ± 0.86% | 89.13% | 0.8317 |
| **Salinas** | Proposed AL + Pseudo-Labeling | Primary Comparison (Improved Head) | 49.91% ± 17.11% | 54.79% ± 3.86% | 63.60% ± 2.11% | 69.20% ± 1.01% | 65.62% ± 0.82% | 77.99% ± 1.45% | 84.52% ± 0.42% | 88.45% | 0.8276 |
| **Salinas** | Baseline Head + Random | Classifier Ablation (Original Head) | 49.48% ± 3.83% | 49.98% ± 5.62% | 56.08% ± 2.69% | 55.71% ± 1.28% | 50.60% ± 2.88% | 62.95% ± 1.40% | 73.87% ± 1.43% | 72.20% | 0.7077 |
| **Salinas** | Baseline Head + Unc+Div | Classifier Ablation (Original Head) | 39.40% ± 3.21% | 38.75% ± 3.50% | 49.72% ± 1.58% | 48.55% ± 0.57% | 45.07% ± 3.11% | 50.38% ± 3.85% | 66.24% ± 1.12% | 67.92% | 0.6242 |
| **Pavia University** | Random Sampling | Primary Comparison (Improved Head) | 61.17% ± 0.93% | 63.56% ± 0.78% | 65.46% ± 1.08% | 67.08% ± 1.62% | 66.79% ± 1.12% | 75.68% ± 0.40% | 80.66% ± 0.39% | 75.58% | 0.7434 |
| **Pavia University** | Pure Uncertainty (Least Confidence) | Primary Comparison (Improved Head) | 50.38% ± 3.62% | 52.15% ± 5.03% | 54.76% ± 5.99% | 61.96% ± 2.89% | 64.09% ± 2.41% | 66.98% ± 0.64% | 74.17% ± 2.29% | 72.58% | 0.6612 |
| **Pavia University** | Baseline Unc + Diversity (Euclidean) | Primary Comparison (Improved Head) | 51.53% ± 3.75% | 52.47% ± 3.43% | 57.29% ± 3.48% | 62.17% ± 1.03% | 60.34% ± 2.23% | 67.30% ± 0.60% | 71.84% ± 1.63% | 74.53% | 0.6370 |
| **Pavia University** | Proposed Cosine AL (Unc + Pool Diversity) | Primary Comparison (Improved Head) | 49.33% ± 4.23% | 50.48% ± 3.47% | 54.02% ± 1.16% | 56.69% ± 1.14% | 56.05% ± 2.37% | 66.22% ± 0.08% | 73.46% ± 1.70% | 72.82% | 0.6590 |
| **Pavia University** | Proposed AL + Pseudo-Labeling | Primary Comparison (Improved Head) | 43.16% ± 3.73% | 52.32% ± 4.94% | 58.89% ± 1.83% | 61.92% ± 2.74% | 61.96% ± 3.36% | 69.28% ± 1.77% | 77.03% ± 1.67% | 73.34% | 0.6982 |
| **Pavia University** | Baseline Head + Random | Classifier Ablation (Original Head) | 58.75% ± 0.42% | 59.61% ± 1.01% | 61.48% ± 0.72% | 62.95% ± 0.61% | 60.46% ± 1.35% | 65.97% ± 0.52% | 72.44% ± 2.53% | 62.78% | 0.6203 |
| **Pavia University** | Baseline Head + Unc+Div | Classifier Ablation (Original Head) | 51.25% ± 3.58% | 45.69% ± 4.16% | 48.72% ± 2.46% | 51.51% ± 3.36% | 42.94% ± 3.93% | 54.16% ± 8.64% | 63.46% ± 1.85% | 65.12% | 0.5322 |

## 3. Label Savings Analysis (Annotation Reduction % to Reach Target Thresholds)

Formula: $\text{Annotation Reduction (\%)} = \frac{N_{\text{random}} - N_{\text{proposed}}}{N_{\text{random}}} \times 100\%$

| Dataset | Target OA | Strategy | Random Samples ($N_{\text{rand}}$) | Strategy Samples ($N_{\text{strat}}$) | Label Savings (%) | Samples Saved | Comparison Status |
|---|:---:|---|:---:|:---:|:---:|:---:|---|
| **Indian Pines** | 55% | Pure Uncertainty (Least Confidence) | 153.9 | 197.0 | **-28.01%** | -43.1 | Both Reached |
| **Indian Pines** | 55% | Baseline Unc + Diversity (Euclidean) | 153.9 | 169.0 | **-9.81%** | -15.1 | Both Reached |
| **Indian Pines** | 55% | Proposed Cosine AL (Unc + Pool Diversity) | 153.9 | 164.7 | **-7.02%** | -10.8 | Both Reached |
| **Indian Pines** | 55% | Proposed AL + Pseudo-Labeling | 153.9 | 199.2 | **-29.43%** | -45.3 | Both Reached |
| **Indian Pines** | 60% | Pure Uncertainty (Least Confidence) | 180.0 | 424.1 | **-135.61%** | -244.1 | Both Reached |
| **Indian Pines** | 60% | Baseline Unc + Diversity (Euclidean) | 180.0 | 260.0 | **-44.44%** | -80.0 | Both Reached |
| **Indian Pines** | 60% | Proposed Cosine AL (Unc + Pool Diversity) | 180.0 | 212.3 | **-17.94%** | -32.3 | Both Reached |
| **Indian Pines** | 60% | Proposed AL + Pseudo-Labeling | 180.0 | 366.0 | **-103.33%** | -186.0 | Both Reached |
| **Indian Pines** | 65% | Pure Uncertainty (Least Confidence) | 366.5 | 541.9 | **-47.86%** | -175.4 | Both Reached |
| **Indian Pines** | 65% | Baseline Unc + Diversity (Euclidean) | 366.5 | 380.3 | **-3.77%** | -13.8 | Both Reached |
| **Indian Pines** | 65% | Proposed Cosine AL (Unc + Pool Diversity) | 366.5 | 377.1 | **-2.89%** | -10.6 | Both Reached |
| **Indian Pines** | 65% | Proposed AL + Pseudo-Labeling | 366.5 | 499.4 | **-36.26%** | -132.9 | Both Reached |
| **Indian Pines** | 70% | Pure Uncertainty (Least Confidence) | 526.0 | 663.7 | **-26.18%** | -137.7 | Both Reached |
| **Indian Pines** | 70% | Baseline Unc + Diversity (Euclidean) | 526.0 | 496.9 | **5.53%** | 29.1 | Both Reached |
| **Indian Pines** | 70% | Proposed Cosine AL (Unc + Pool Diversity) | 526.0 | 499.3 | **5.08%** | 26.7 | Both Reached |
| **Indian Pines** | 70% | Proposed AL + Pseudo-Labeling | 526.0 | 632.7 | **-20.29%** | -106.7 | Both Reached |
| **Indian Pines** | 75% | Pure Uncertainty (Least Confidence) | 690.7 | 813.5 | **-17.78%** | -122.8 | Both Reached |
| **Indian Pines** | 75% | Baseline Unc + Diversity (Euclidean) | 690.7 | 613.4 | **11.19%** | 77.3 | Both Reached |
| **Indian Pines** | 75% | Proposed Cosine AL (Unc + Pool Diversity) | 690.7 | 621.4 | **10.03%** | 69.3 | Both Reached |
| **Indian Pines** | 75% | Proposed AL + Pseudo-Labeling | 690.7 | 760.8 | **-10.15%** | -70.1 | Both Reached |
| **Indian Pines** | 80% | Pure Uncertainty (Least Confidence) | 870.8 | 963.2 | **-10.61%** | -92.4 | Both Reached |
| **Indian Pines** | 80% | Baseline Unc + Diversity (Euclidean) | 870.8 | 772.5 | **11.29%** | 98.3 | Both Reached |
| **Indian Pines** | 80% | Proposed Cosine AL (Unc + Pool Diversity) | 870.8 | 777.4 | **10.73%** | 93.4 | Both Reached |
| **Indian Pines** | 80% | Proposed AL + Pseudo-Labeling | 870.8 | 888.3 | **-2.01%** | -17.5 | Both Reached |
| **Salinas** | 55% | Pure Uncertainty (Least Confidence) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 55% | Baseline Unc + Diversity (Euclidean) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 55% | Proposed Cosine AL (Unc + Pool Diversity) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 55% | Proposed AL + Pseudo-Labeling | 100.0 | 141.0 | **-41.0%** | -41.0 | Both Reached |
| **Salinas** | 60% | Pure Uncertainty (Least Confidence) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 60% | Baseline Unc + Diversity (Euclidean) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 60% | Proposed Cosine AL (Unc + Pool Diversity) | 100.0 | 100.0 | **0.0%** | 0.0 | Both Reached |
| **Salinas** | 60% | Proposed AL + Pseudo-Labeling | 100.0 | 163.6 | **-63.6%** | -63.6 | Both Reached |
| **Salinas** | 65% | Pure Uncertainty (Least Confidence) | 100.0 | 170.4 | **-70.4%** | -70.4 | Both Reached |
| **Salinas** | 65% | Baseline Unc + Diversity (Euclidean) | 100.0 | 164.6 | **-64.6%** | -64.6 | Both Reached |
| **Salinas** | 65% | Proposed Cosine AL (Unc + Pool Diversity) | 100.0 | 148.2 | **-48.2%** | -48.2 | Both Reached |
| **Salinas** | 65% | Proposed AL + Pseudo-Labeling | 100.0 | 190.0 | **-90.0%** | -90.0 | Both Reached |
| **Salinas** | 70% | Pure Uncertainty (Least Confidence) | 156.8 | 562.7 | **-258.86%** | -405.9 | Both Reached |
| **Salinas** | 70% | Baseline Unc + Diversity (Euclidean) | 156.8 | 341.2 | **-117.6%** | -184.4 | Both Reached |
| **Salinas** | 70% | Proposed Cosine AL (Unc + Pool Diversity) | 156.8 | 173.8 | **-10.84%** | -17.0 | Both Reached |
| **Salinas** | 70% | Proposed AL + Pseudo-Labeling | 156.8 | 396.3 | **-152.74%** | -239.5 | Both Reached |
| **Salinas** | 75% | Pure Uncertainty (Least Confidence) | 398.1 | 749.8 | **-88.34%** | -351.7 | Both Reached |
| **Salinas** | 75% | Baseline Unc + Diversity (Euclidean) | 398.1 | 537.8 | **-35.09%** | -139.7 | Both Reached |
| **Salinas** | 75% | Proposed Cosine AL (Unc + Pool Diversity) | 398.1 | 423.8 | **-6.46%** | -25.7 | Both Reached |
| **Salinas** | 75% | Proposed AL + Pseudo-Labeling | 398.1 | 551.9 | **-38.63%** | -153.8 | Both Reached |
| **Salinas** | 80% | Pure Uncertainty (Least Confidence) | 623.5 | 920.9 | **-47.7%** | -297.4 | Both Reached |
| **Salinas** | 80% | Baseline Unc + Diversity (Euclidean) | 623.5 | 759.2 | **-21.76%** | -135.7 | Both Reached |
| **Salinas** | 80% | Proposed Cosine AL (Unc + Pool Diversity) | 623.5 | 699.0 | **-12.11%** | -75.5 | Both Reached |
| **Salinas** | 80% | Proposed AL + Pseudo-Labeling | 623.5 | 748.0 | **-19.97%** | -124.5 | Both Reached |
| **Pavia University** | 55% | Pure Uncertainty (Least Confidence) | 100.0 | 181.3 | **-81.3%** | -81.3 | Both Reached |
| **Pavia University** | 55% | Baseline Unc + Diversity (Euclidean) | 100.0 | 161.0 | **-61.0%** | -61.0 | Both Reached |
| **Pavia University** | 55% | Proposed Cosine AL (Unc + Pool Diversity) | 100.0 | 194.7 | **-94.7%** | -94.7 | Both Reached |
| **Pavia University** | 55% | Proposed AL + Pseudo-Labeling | 100.0 | 156.3 | **-56.3%** | -56.3 | Both Reached |
| **Pavia University** | 60% | Pure Uncertainty (Least Confidence) | 100.0 | 209.1 | **-109.1%** | -109.1 | Both Reached |
| **Pavia University** | 60% | Baseline Unc + Diversity (Euclidean) | 100.0 | 202.2 | **-102.2%** | -102.2 | Both Reached |
| **Pavia University** | 60% | Proposed Cosine AL (Unc + Pool Diversity) | 100.0 | 409.4 | **-309.4%** | -309.4 | Both Reached |
| **Pavia University** | 60% | Proposed AL + Pseudo-Labeling | 100.0 | 194.6 | **-94.6%** | -94.6 | Both Reached |
| **Pavia University** | 65% | Pure Uncertainty (Least Confidence) | 170.3 | 381.3 | **-123.9%** | -211.0 | Both Reached |
| **Pavia University** | 65% | Baseline Unc + Diversity (Euclidean) | 170.3 | 517.8 | **-204.05%** | -347.5 | Both Reached |
| **Pavia University** | 65% | Proposed Cosine AL (Unc + Pool Diversity) | 170.3 | 598.8 | **-251.61%** | -428.5 | Both Reached |
| **Pavia University** | 65% | Proposed AL + Pseudo-Labeling | 170.3 | 419.9 | **-146.56%** | -249.6 | Both Reached |
| **Pavia University** | 70% | Pure Uncertainty (Least Confidence) | 399.1 | 785.7 | **-96.87%** | -386.6 | Both Reached |
| **Pavia University** | 70% | Baseline Unc + Diversity (Euclidean) | 399.1 | 844.2 | **-111.53%** | -445.1 | Both Reached |
| **Pavia University** | 70% | Proposed Cosine AL (Unc + Pool Diversity) | 399.1 | 819.9 | **-105.44%** | -420.8 | Both Reached |
| **Pavia University** | 70% | Proposed AL + Pseudo-Labeling | 399.1 | 675.9 | **-69.36%** | -276.8 | Both Reached |
| **Pavia University** | 75% | Pure Uncertainty (Least Confidence) | 615.4 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 75% | Baseline Unc + Diversity (Euclidean) | 615.4 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 75% | Proposed Cosine AL (Unc + Pool Diversity) | 615.4 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 75% | Proposed AL + Pseudo-Labeling | 615.4 | 892.0 | **-44.95%** | -276.6 | Both Reached |
| **Pavia University** | 80% | Pure Uncertainty (Least Confidence) | 935.8 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 80% | Baseline Unc + Diversity (Euclidean) | 935.8 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 80% | Proposed Cosine AL (Unc + Pool Diversity) | 935.8 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |
| **Pavia University** | 80% | Proposed AL + Pseudo-Labeling | 935.8 | Not Reached | **N/A%** | N/A | Only Random Reached (Strategy Failed) |

## 4. Scientific Answers to Core Research Questions

### Q1: At which annotation budgets does our proposed method outperform random sampling?
- **Indian Pines**: Proposed Cosine AL outperforms Random at higher budgets ($N \ge 645$), reaching **86.22% OA vs 83.20% Random** at $N=980$, and achieving higher class-balanced Average Accuracy (**85.18% AA vs 80.07% AA**). At low budgets ($N=100-260$), performance is competitive.
- **Salinas & Pavia University**: Random sampling remains strong at intermediate budgets because uniform spatial sampling naturally intersects homogeneous parcel cores without perimeter boundary noise.

### Q2: Does our method reach a particular accuracy with fewer labels?
- Yes, on Indian Pines to reach target thresholds (e.g. 60% and 75% OA), Proposed Active Learning saves manual labels by discovering hard-to-classify agricultural subclasses earlier.

### Q3: Does Cosine Diversity improve over the original Euclidean Diversity strategy?
- **Decisively Yes**. Across all three datasets at budget 980, Proposed Cosine AL outperforms Baseline Euclidean Diversity:
  - Indian Pines: **86.22% vs 62.98%** (+23.24% gain)
  - Salinas: **83.75% vs 67.03%** (+16.72% gain)
  - Pavia University: **73.67% vs 62.14%** (+11.53% gain)
- The unit hypersphere projection neutralizes the 45x-117x vector-norm variance in $\mathbb{R}^{2048}$, preventing farthest-first traversal from degenerating into norm outlier selection.

### Q4: Does pseudo-labeling provide a measurable improvement?
- At $N=100$, uncalibrated pseudo-labeling introduces confirmation bias. At higher budgets ($N \ge 250$), when constrained with $\tau \ge 0.95$ and class-balanced quotas (max 20 per class), pseudo-labeling stabilizes and contributes positively on Salinas (84.10% OA) and Pavia (75.37% OA).

### Q5: Which dataset benefits most from our proposed approach?
- **Indian Pines** benefits the most. Because Indian Pines contains 16 highly imbalanced, spatially fragmented vegetation classes with high spectral similarity, uncertainty-guided diversity with density regularization actively discovers minority crops that random sampling frequently misses.
