# ConformAL-HSI: Publication Claims Taxonomy

**Module:** Master Research Audit & Reconciliation  
**Date:** November 2026  
**Status:** Scientifically Defensible  

This document categorizes all potential research and publication claims for the ConformAL-HSI project into four strict scientific tiers, establishing an unassailable empirical foundation for academic manuscripts.

---

## Tier 1: Claims Directly Supported by Experimental Evidence

The following claims are established by rigorous empirical measurements across multi-seed evaluations and paired statistical tests:

1. **Margin Sampling + Cosine Diversity Decisively Outperforms Original LC + FD:**
   - Across 150 matched post-initial evaluation pairs (rounds 1–10 across 3 datasets and 5 seeds), Margin + Cosine achieves a **73.33% win rate** over Original LC + FD (110 wins, 0 ties, 40 losses).
   - The overall mean paired accuracy difference is **$+1.891\% \pm 3.554\%$ OA** (95% CI: $[+1.323\%, +2.460\%]$).
   - Paired differences are highly statistically significant: paired $t$-test $t = 6.5186$, $p = 1.03 \times 10^{-9}$; Wilcoxon signed-rank test $W = 2565.5$, $p = 6.22 \times 10^{-9}$.

2. **Dataset-Specific Statistical Superiority:**
   - **Salinas:** Margin + Cosine outperforms LC + FD by **$+2.778\% \pm 3.566\%$ OA** across rounds 1–10 (76.0% win rate, $t = 5.508$, $p = 1.33 \times 10^{-6}$). Margin + Cosine reaches the 60% OA milestone at **$N=220$**, saving 20 labels (**8.33% discrete label reduction**) relative to LC + FD ($N=240$).
   - **Indian Pines:** Margin + Cosine outperforms LC + FD by **$+1.945\% \pm 3.473\%$ OA** across rounds 1–10 (78.0% win rate, $t = 3.959$, $p = 2.43 \times 10^{-4}$).
   - **Pavia University:** Margin + Cosine reaches the 60% OA threshold at **$N=290$** ($60.44\%$), whereas Original LC + FD **fails to cross 60% OA** within the evaluated range ($N \le 290$, peaking at $59.85\%$).

3. **Empirical Elimination of Catastrophic Active Learning Degradation:**
   - LC + FD exhibits severe single-seed degradation (e.g., dropping to 49.74% on Indian Pines Seed 43 and 56.41% on Pavia Seed 42 due to outlier trapping). Margin + Cosine completely stabilizes training, raising minimum worst-case seed performance by up to +6.0% OA.

4. **Classifier Head Architecture Dramatically Dictates Probing Accuracy:**
   - Upgrading from the under-parameterized Head A (256 hidden units, LayerNorm, no dropout) to Head B (512 units, BatchNorm1d, Dropout 0.1) increases classification accuracy across all budgets by **+13.26% OA at $N=220$ on Salinas** and enables active learning to decisively surpass Random Sampling.

---

## Tier 2: Claims Supported with Explicit Qualifications

The following claims are valid only when stated with appropriate qualifiers and methodological disclosures:

1. **"Active Learning Saves Annotations to Reach 60% Overall Accuracy":**
   - *Qualification:* Supported on **Salinas** under Head A, where Margin + Cosine saves 20 labels (8.33% savings) relative to LC + FD and matches Random. Under Head B, Margin + Cosine achieves 74.60% OA at $N=220$.
   - *Limitation:* On Pavia University and Indian Pines under the constrained Head A, Random Sampling crosses 60% OA earlier than active learning because Random benefits from class-proportional sampling of dominant background classes (Meadows on Pavia, Soybean on IP), while active learning shifts queries to ambiguous minority boundaries that low-capacity heads struggle to fit without underfitting.

2. **"Sub-Second Active Learning Query and Retraining Latency":**
   - *Qualification:* Valid because training operates on pre-extracted, frozen 2048-dimensional embeddings using a lightweight neural head (~0.53M to ~1.19M parameters). Retraining takes **$0.021 \text{ s}$ to $0.052 \text{ s}$** per round, and acquisition takes **$0.015 \text{ s}$ to $0.045 \text{ s}$**, enabling instantaneous human-in-the-loop interaction.
   - *Limitation:* This computational efficiency applies strictly to inference on cached embeddings; extracting the initial backbone features from raw hyperspectral cubes requires GPU preprocessing.

3. **"Label Efficiency Across Hyperspectral Datasets":**
   - *Qualification:* Valid within the scope of our evaluated random pixel-split partitions (70% candidate pool, 15% validation pool, 15% test pool) on Indian Pines, Salinas, and Pavia University.
   - *Limitation:* The degree of label efficiency depends heavily on class balance; datasets with extreme dominant classes require class-balanced query constraints or higher classifier capacity to prevent dominant-class starvation.

---

## Tier 3: Hypotheses Requiring Further Experimental Verification

The following statements represent plausible physical mechanisms that are supported by preliminary correlations but require dedicated follow-up experiments:

1. **Boundary-Sample Underfitting Mechanism in Low-Capacity Heads:**
   - *Observation:* Training loss under Head A escalates monotonically as active learning queries boundary samples (rising from 0.80 to 1.17 on IP for LC+FD), while Random loss remains flat at 0.88.
   - *Hypothesis:* Low-capacity classifier heads without dropout lack the representational flexibility to separate dense clusters of ambiguous boundary queries without overfitting the acquired points and sacrificing dominant-class generalization.
   - *Required Verification:* Synthetic controlled 2D manifold experiments and grad-CAM / feature-attribution analysis on decision boundary samples.

2. **Euclidean Outlier Trapping in Hyperspectral Embeddings:**
   - *Observation:* Euclidean Farthest-First traversal on unnormalized embeddings occasionally selects uninformative isolated points.
   - *Hypothesis:* In 2048-dimensional feature spaces, unnormalized Euclidean distance is dominated by feature magnitude rather than semantic class separation, whereas cosine similarity on unit-normalized embeddings preserves angular class separation.
   - *Required Verification:* Spectral radius and nearest-neighbor density distributions of selected query points.

---

## Tier 4: Claims That Must NOT Be Made (Scientifically Insupportable)

The following claims are contradicted by the empirical data or lack methodological basis, and must **never** appear in papers or presentations:

1. **DO NOT CLAIM that ConformAL-HSI currently incorporates Conformal Prediction:**
   - *Fact:* No split-conformal prediction, prediction sets, nonconformity scores, or conformal coverage guarantees are implemented in the active learning loop. The framework currently implements uncertainty- and diversity-guided active learning with temperature scaling.

2. **DO NOT CLAIM that Margin + Cosine beats Random Sampling at every budget under Head A:**
   - *Fact:* On Pavia University, Random Sampling outperforms Margin + Cosine by **6.18% OA on average** ($p = 4.26 \times 10^{-20}$), reaching 60% OA at $N=170$ versus $N=290$ for Margin + Cosine. On Indian Pines, Random reaches 60% at $N=360$, while Margin + Cosine peaks at 57.30%.

3. **DO NOT CLAIM an 88% Paired Win Rate across All Seeds:**
   - *Fact:* The exact post-initial paired win rate across 150 comparisons is **73.33%**.

4. **DO NOT CLAIM that ConformAL-HSI beats 98% published literature models (e.g., HybridSN, SpectralFormer) at low budgets:**
   - *Fact:* Published papers evaluated models with 10%–30% training data (1,000–5,000+ samples). At high budgets ($N=2,000$), our frozen probe reaches 92.69% on IP, 89.20% on Salinas, and 86.76% on Pavia, while end-to-end models reach ~98% because their backbone convolutions adapt fully to domain-specific spectra.

5. **DO NOT CLAIM that the evaluation proves spatial generalization across geographic regions:**
   - *Fact:* The data splits use random pixel coordinate partitioning with $15 \times 15$ patches, which permits spatial autocorrelation between overlapping patches in train and test sets. Spatial generalization requires a spatially disjoint block split.

6. **DO NOT CLAIM that the split was 10% candidate / 10% val / 80% test:**
   - *Fact:* The actual cached data tensors used in all experiments are **70% candidate pool / 15% validation pool / 15% test pool**.

