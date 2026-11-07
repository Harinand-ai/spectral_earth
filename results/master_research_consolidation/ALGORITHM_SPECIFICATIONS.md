# ConformAL-HSI: Acquisition Algorithm Specifications

**Document:** Mathematical & Algorithmic Method Specifications  
**Date:** November 2026  
**Status:** Validated & Audited  

This document details the exact mathematical formulas, algorithmic steps, hyperparameters, computational complexity, and failure modes for all acquisition strategies evaluated in the ConformAL-HSI framework.

---

## 1. Random Sampling (Baseline)
- **Formula:** $P(x_i \text{ selected}) = \frac{1}{|\mathcal{U}|}$.
- **Algorithm:** Draw $k$ uniform random samples without replacement from the unlabeled pool: `rng.choice(len(unlabeled), size=k, replace=False)`.
- **Complexity:** $\mathcal{O}(k)$ time, $\mathcal{O}(1)$ memory.
- **Strengths:** Draws proportionally from true underlying class distributions; samples representative cluster interiors.
- **Limitations:** Completely unguided; queries highly redundant points in dense interior regions.

---

## 2. Original Strategy: Least Confidence + Farthest-First Diversity (LC + FD)
- **Mathematical Formula:**
  1. *Uncertainty:* $u_{\text{lc}}(x) = 1 - \max_{c \in \{1, \dots, C\}} p(y=c \mid x)$.
  2. *Candidate Filter:* $\mathcal{C} = \text{Top}_{4k}(-u_{\text{lc}})$.
  3. *Euclidean Farthest-First Traversal on Raw Features:*
     $$x_1 = \arg\max_{x \in \mathcal{C}} u_{\text{lc}}(x), \quad x_m = \arg\max_{x \in \mathcal{C} \setminus \mathcal{S}_{m-1}} \min_{s \in \mathcal{S}_{m-1}} \| z(x) - z(s) \|_2$$
- **Complexity:** $\mathcal{O}(N_u C)$ inference + $\mathcal{O}(k \cdot |\mathcal{C}| \cdot D)$ traversal.
- **Failure Mode & Vulnerability:** Cached features have Euclidean norms on the order of $\approx 16,850$. Unnormalized Euclidean distance allows isolated feature outliers with extreme norms to dominate distance calculations, trapping the acquisition set in uninformative noise regions.

---

## 3. Pure Margin Sampling
- **Mathematical Formula:**
  $$u_{\text{margin}}(x) = 1 - \left(p_{(1)}(x) - p_{(2)}(x)\right)$$
  where $p_{(1)}(x)$ and $p_{(2)}(x)$ are the largest and second largest predicted class probabilities.
- **Selection:** $\mathcal{S} = \arg\max_{x \in \mathcal{U}}^{(k)} u_{\text{margin}}(x)$.
- **Complexity:** $\mathcal{O}(N_u C \log C)$ probability sorting + $\mathcal{O}(N_u \log k)$ selection.
- **Strengths:** Directly targets genuine two-class decision boundaries. Highly optimization-friendly for low-capacity neural classifiers.
- **Limitations:** Lacks explicit spatial/feature diversity; can select tightly clustered points along the same boundary.

---

## 4. Margin + Cosine Diversity
- **Mathematical Formula:**
  1. *Uncertainty Filter:* $\mathcal{C} = \text{Top}_{\max(4k, 200)} u_{\text{margin}}(x)$ on unit-normalized embeddings ($\|z\|_2 = 1$).
  2. *Cosine Diversity to Labeled Pool & Batch:*
     $$d_{\cos}(x, \mathcal{L} \cup \mathcal{S}_m) = 1 - \max_{z \in \mathcal{L} \cup \mathcal{S}_m} x^\top z$$
     Iteratively select points maximizing minimum cosine distance:
     $$x_m = \arg\max_{x \in \mathcal{C} \setminus \mathcal{S}_{m-1}} \min_{z \in \mathcal{L} \cup \mathcal{S}_{m-1}} (1 - x^\top z)$$
- **Complexity:** $\mathcal{O}(N_u C)$ + $\mathcal{O}(|\mathcal{C}| \cdot |\mathcal{L}| \cdot D) + \mathcal{O}(k \cdot |\mathcal{C}| \cdot D)$.
- **Strengths:** Eliminates Euclidean outlier trapping; ensures boundary queries are non-redundant and dispersed relative to the existing training pool.

---

## 5. Conformal Uncertainty Only
- **Mathematical Formula:**
  Using calibrated nonconformity quantile $\hat{q}$ on validation pool:
  $$\mathcal{C}(x) = \{c : \hat{p}(y=c \mid x) \ge 1 - \hat{q}\}$$
  $$u_{\text{conf}}(x) = 0.6 \cdot \frac{|\mathcal{C}(x)| - 1}{\max(1, C - 1)} + 0.4 \cdot \left(1 - \text{margin}(\mathcal{C}(x))\right)$$
- **Selection:** Top $k$ samples maximizing $u_{\text{conf}}(x)$.
- **Strengths:** Distribution-free uncertainty measure directly reflecting prediction set ambiguity.
- **Limitations:** Prioritizes multi-class overlap regions (set sizes $\ge 4$) that require higher classifier capacity to resolve.

---

## 6. Conformal + Cosine Diversity (Proposed ConformAL-HSI)
- **Mathematical Formula:**
  Combines Conformal Prediction Set Cardinality with Cosine Farthest-First Pool Diversity:
  1. Top candidate filtering using $u_{\text{conf}}(x)$ on $\mathcal{U}$.
  2. Cosine Farthest-First Traversal relative to $\mathcal{L}$ and accumulating batch $\mathcal{S}$.
- **Complexity:** Identical to Margin + Cosine.
- **Properties:** Provides finite-sample marginal coverage guarantees on test data while querying geographically and semantically dispersed points of provable model ambiguity.

---

## 7. Conformal + Cosine + Class Coverage
- **Mathematical Formula:**
  $$s_{\text{adj}}(x) = u_{\text{conf}}(x) \cdot \sum_{c=1}^C \hat{p}(y=c \mid x) \cdot \min\left(2.0, \; \left(\frac{\pi_c}{f_c}\right)^{\lambda_{\text{cov}}}\right)$$
  where $\pi_c$ is the candidate pool class prior and $f_c$ is the current labeled pool class proportion.
- **Strengths:** Dynamically prevents dominant-class starvation (e.g. Meadows on Pavia) and rare-class neglect without accessing test labels.
