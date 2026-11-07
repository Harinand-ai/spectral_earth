# ConformAL-HSI: Master Research Claims Taxonomy

**Document:** Formal Scientific Claims Taxonomy  
**Date:** November 2026  
**Status:** Scientifically Defensible  

---

## Tier 1: Claims Directly Supported by Experimental Evidence

1. **Margin Sampling Reaches Fixed Overall Accuracy Milestones with Substantially Fewer Labels on Salinas:**
   - On Salinas under Head A, Pure Margin Sampling reaches 60% OA at **240 labels** ($61.08\%$), saving 60 labels (**20.00% discrete label reduction**) relative to Random Sampling ($N=300$, $60.91\%$).
   - At the 55% OA milestone, Margin reaches the threshold at **180 labels** ($55.18\%$) vs 220 labels for Random (**18.18% label reduction**).
2. **Margin Sampling Outperforms Original LC + FD across All Datasets:**
   - Statistically significant seed-level AUCL superiority on Indian Pines ($p = 0.044$), Salinas ($p = 0.0009$), and Pavia University ($p = 0.0002$).
   - Crushes LC + FD on final Overall Accuracy by **$+16.77\%$ on Pavia University** ($67.52\%$ vs $50.76\%$), **$+6.17\%$ on Salinas** ($62.38\%$ vs $56.21\%$), and **$+7.66\%$ on Indian Pines** ($57.27\%$ vs $49.61\%$).
3. **Split Conformal Prediction Delivers Exact Empirical Marginal Coverage:**
   - Across all 1,155 evaluations, empirical coverage on held-out test data precisely aligns with the nominal $90\%$ target ($90.10\% \pm 0.38\%$), validating mathematical calibration on validation features.
4. **Sub-Second Real-Time Active Learning Latency:**
   - Retraining on pre-extracted foundation embeddings takes **$0.025 \text{ s}$ to $0.050 \text{ s}$** per round; acquisition takes **$0.015 \text{ s}$ to $0.045 \text{ s}$**, enabling instantaneous human-in-the-loop querying.

---

## Tier 2: Claims Supported with Explicit Qualifications

1. **"Margin Sampling Outperforms Random Sampling":**
   - *Supported on Salinas* (saving 20% labels to reach 60% OA) and *Pavia University* (reaching $67.52\%$ vs $65.54\%$ final OA).
   - *Limitation:* On Indian Pines, final OA is comparable ($57.27\%$ vs $56.53\%$, $+0.74\%$ gain, $p = 0.44$).
2. **"Conformal Active Learning Provides Principled Sample Selection":**
   - *Supported:* Conformal prediction sets provide rigorous distribution-free uncertainty and shrink as sample budgets grow.
   - *Limitation:* Querying based solely on prediction set cardinality $|\mathcal{C}(x)|$ prioritizes ambiguous multi-class overlap regions that low-capacity heads struggle to separate, making pure two-class Margin Sampling more optimization-friendly.
3. **"Label Efficiency Across Hyperspectral Scenes":**
   - *Supported within the evaluated random pixel partitions (70/15/15 split).*
   - *Limitation:* Random-pixel partitioning exhibits spatial autocorrelation leakage; absolute accuracy is lower when evaluated under a spatially disjoint block partition.

---

## Tier 3: Hypotheses Requiring Further Verification

1. **High-Capacity Heads Unlock Conformal Multi-Class Querying:**
   - *Hypothesis:* Set-cardinality conformal queries (selecting points where $|\mathcal{C}(x)| \ge 4$) underperform two-class margin queries under Head A because a 256-unit MLP underfits complex multi-class boundaries. Under Head B (512 units + BatchNorm + Dropout), conformal queries are hypothesized to surpass Margin Sampling.
2. **True Cross-Scene Geographic Generalization:**
   - *Hypothesis:* Spatial foundation models pretrained on EnMAP/DESIS transfer better to cross-region scenes than models trained from scratch on single scenes.

---

## Tier 4: Claims That Must NOT Be Made (Scientifically Insupportable)

1. **DO NOT CLAIM that ConformAL-HSI beats 98% published literature models (HybridSN, SpectralFormer):**
   - Published models used **3,000 to 5,000+ labels** (10% to 30% of entire datasets). When probed at $N=2,000$, our frozen probe reaches 92.69% on IP and 89.20% on Salinas. Probing frozen features does not reach 98% because the backbone weights are not adapted to specific sensor channels.
2. **DO NOT CLAIM that the split was 10% candidate / 10% val / 80% test:**
   - The actual cached split used across all benchmarks is **70% candidate pool / 15% validation pool / 15% test pool**.
3. **DO NOT CLAIM that the evaluation proves spatial generalization across geographic regions:**
   - The random pixel split has 100% patch overlap. Spatially disjoint block partitions are required to claim geographic holdout generalization.
4. **DO NOT CLAIM an 88% paired win rate from correlated checkpoints:**
   - Re-analysis at the independent seed level proves Margin beats LC+FD with $p < 0.05$ across all datasets via independent AUCL testing.
