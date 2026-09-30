# MODEL IMPROVEMENT AUDIT
## SpectralEarth Active Learning Framework — Independent Technical Review

**Auditor role**: Independent Senior Hyperspectral Image Classification + Active Learning Researcher
**Date**: 2026-08-13
**Codebase inspected**:
- `src/backbones/spec_resnet.py`, `spectral_adapter.py`
- `src/learning_engine/classifier.py`, `feature_extractor.py`, `trainer.py`, `learning_engine.py`, `dataset_manager.py`
- `src/query_engine/query_engine/engine.py`, `uncertainty.py`, `diversity.py`, `adaptive.py`
- `src/datasets/indian_pines.py`, datamodules
- `src/utils/spectral_adapter_1d.py`
- `run_multiseed_validation.py`, `run_cross_sensor_eval.py`
- `FINAL_PROJECT_STATUS.md`, `RESEARCH_CLAIMS.md`, `CROSS_SENSOR_AUDIT.md`

---

## 1. Current Architecture Assessment

### Architecture Trace

```
Input patch [B, C, 15, 15]   (C = 200/204/103 bands)
      |
SpectralAdapter (pretrained, frozen)
  3x Conv3d -> BN3d -> ReLU
  AdaptiveAvgPool3d(1, H, W)
-> [B, 128, 15, 15]
      |
CustomResNet50 (pretrained, frozen)
  layer1 -> layer2 -> layer3 -> layer4
  global_pool -> flatten
-> [B, 2048]    <- feature vector
      |
ClassifierHead (trainable, ~533K params)
  input_norm: LayerNorm(2048)
  head[0]: Linear(2048 -> 256)
  head[1]: LayerNorm(256)
  head[2]: GELU()
  head[3]: Dropout(p=0.0)   <- DROPOUT IS ZERO
  head[4]: Linear(256 -> C)
-> logits
      |
AdamW + CosineAnnealingLR, 25 epochs/round

QueryEngine (Model B)
  filter_diversity:
    LC uncertainty -> top (4k or 50) candidates
    Greedy farthest-first traversal in L2 embedding space (L2-normalized)
    -> k selected indices
```

---

## 2. Current Strengths

These areas are **already well implemented**. Do not change them without strong justification.

| Area | What is Good |
| :--- | :--- |
| **Backbone freezing** | Strict — `eval()` permanently forced even when `.train()` is called. No gradient leakage. |
| **Class-balanced initial seed** | 10 samples/class using class-stratified sampling. Correct. |
| **Farthest-first diversity** | Greedy k-center in L2-normalized embedding space. Computationally cheap, theoretically sound. |
| **Evaluation metrics** | OA, AA, Macro-F1, Kappa, ECE, Precision, Recall, Confusion Matrix. Comprehensive. |
| **Multi-seed validation** | 5 seeds x 3 strategies x 3 datasets = 45 runs. Provides confidence intervals. |
| **Statistical significance** | Paired t-tests across seeds. Defensible. |
| **Model A / Model B separation** | Clean boundary: only embeddings and probabilities cross the interface. Correct design. |
| **Gradient clipping** | `clip_grad_norm_(..., 1.0)` prevents unstable updates on small labeled pools. |
| **Spectral adapter (backbone)** | 3-layer 1D conv (Conv3d) with adaptive pooling maps arbitrary band counts to 128 channels. This is the actual spectral-awareness of the system. |

---

## 3. Current Weaknesses

### W1 — Dropout is Disabled (p=0.0) on a Very Small Labeled Pool

`ClassifierHead` initializes `Dropout(p=dropout)` and `dropout` defaults to `0.1` in the class
signature, but the calling code in `run_multiseed_validation.py` and `run_cross_sensor_eval.py`
never passes a `dropout` argument. Therefore the actual `dropout` used is `0.0` by default
(checked in classifier.py line 23). **With only 160 labeled samples per dataset
(10/class x 16 classes), this is a significant regularization gap.**

### W2 — Diversity Score Is Unlabeled-Centroid-Based, Not Labeled-Pool-Aware

The diversity score in `engine.py:compute_diversity_scores()` measures distance from the
**centroid of the unlabeled pool**. It does **not** consider the labeled pool at all. A truly
informative diversity criterion would select samples farthest from the already-labeled data,
not from the unlabeled pool centroid. The current approach selects samples at the edges of the
unlabeled distribution, which is a reasonable but suboptimal approximation.

### W3 — filter_diversity Uses Absolute Candidate Pool Size (max(4k, 50)), Not Proportional

```python
cand_pool_size = min(len(embeddings), max(4 * k, 50))
```

With `k=20`, the candidate pool is `max(80, 50) = 80`. On Indian Pines with ~7,000+ unlabeled
samples, this filters to 80/7000 = 1.1% of the pool. Early in training, the classifier is
unreliable, so the top-80 most uncertain samples could be largely noise or outliers. The
farthest-first traversal then selects from 80 potentially unreliable candidates.
A proportional filter (e.g., top 20%) would be more robust.

### W4 — Embedding Normalization Is Applied Inconsistently Across Functions

The farthest-first traversal in `diversity.py:normalize_embeddings()` L2-normalizes embeddings
before computing pairwise distances. The diversity score in `engine.py:compute_diversity_scores()`
also normalizes separately. `select_diverse_samples` normalizes internally while
`compute_diversity_scores()` normalizes externally. This is not wrong but creates code
duplication and confusion.

### W5 — Pavia University: Random Outperforms All AL Strategies on OA

```
Random:            OA = 65.51 +/- 1.96%
LC + FD:           OA = 60.01 +/- 1.54%
Least Confidence:  OA = 55.71 +/- 1.70%
```

This is the most important signal in the results. Active learning is **hurting** Pavia University
on OA. The most likely explanation: Pavia University has a severe class imbalance and
uncertainty-based selection over-queries the minority-class boundary regions, causing the
classifier to overfit to difficult boundary samples at the expense of the majority class.
The model pays in OA to gain AA (AA is 65.97 for LC+FD vs 64.14 for Random).
This tells us: **the current query strategy does not account for class distribution**.

### W6 — Warm-Start Has Not Been Ablated

Every round retrains from the previous head weights (warm-start, reset_head=False). This is
generally good but can cause the classifier to be biased toward patterns in early labeled rounds.
No ablation has confirmed this is better than cold-start for these datasets and this budget.

### W7 — Data Split Has Spatial Correlation Between Train and Test

In `indian_pines.py`, training/validation/test are split pixel-by-pixel using
`np.random.seed(42)` shuffle of all labeled coordinates. This means spatially adjacent pixels
can appear in train and test. Since each sample is a **15x15 patch**, a pixel at position (r, c)
in train and a pixel at position (r+1, c) in test share 14x15 = 210 pixels of overlap.
This is the standard benchmark practice for these datasets, so it is not unique to this
codebase, but it means OA metrics are likely optimistic compared to spatially disjoint evaluation.

### W8 — Uncertainty Estimation Uses a Single Deterministic Forward Pass

All uncertainty measures (LC, Entropy, Margin) use a single softmax output from the classifier.
Since the classifier is a small MLP trained on a very small labeled pool, these probabilities
are **poorly calibrated** (confirmed by ECE being reported). Single-pass uncertainty is known
to be overconfident. This limits the reliability of the query selection signal.

### W9 — Cross-Sensor Transfer Uses Naive 1D Linear Spectral Interpolation

`SpectralResampler1D` uses `F.interpolate(..., mode='linear', align_corners=True)` across the
channel dimension. For sensor transfer (AVIRIS to ROSIS), this maps 200 spectral bands to 103
by linear interpolation, which **does not** account for:
- Different spectral response functions between sensors
- Different wavelength ranges (AVIRIS: 400-2500nm, ROSIS: 430-860nm)
- Atmospheric absorption bands

The adapter is functionally a "pretend the spectra are smooth and evenly spaced" approximation.
The performance drop on cross-sensor Pavia pairs (49.84%, 54.12%) vs within-dataset Pavia
(65.51%) confirms this limitation.

### W10 — Initial Classifier Training Uses Same LR/Epochs as AL Rounds

Round 0 (initial classifier on seed pool, 10 samples/class) uses the same `lr=0.0005`,
`epochs=25` as subsequent rounds with 20+ additional labeled samples. Early rounds with very
few samples might benefit from more conservative settings.

### W11 — Evaluation Does Not Track Per-Round Label Efficiency (AULC)

The history is logged per-round, but no **Area Under the Learning Curve (AULC)** is computed
or reported. AULC is a more complete measure of label efficiency than final OA alone.
A strategy that reaches 60% at round 2 and plateaus is less label-efficient than one that
reaches 60% at round 4 but with a steeper slope.

### W12 — No Spatial Diversity Component

The 15x15 patch input means adjacent pixels have highly correlated feature embeddings.
The query engine selects k=20 samples per round purely based on embedding-space diversity.
It is possible that many selected samples come from the same spatial region (e.g., a single
homogeneous field in Salinas), providing redundant information.
**No spatial distance constraint exists.**

### W13 — Cross-Sensor Transfer Bug Corrected But Mechanism Is Still Shallow

The corrected transfer (FC1 weight copy + spectral resampling) improves over pure within-dataset
AL on same-sensor transfers (IP->Sal: 74.14% vs Salinas within: 72.19%) but underperforms on
cross-sensor (IP->Pavia: 49.84%). The transfer mechanism relies only on the first hidden layer
of the classifier and naive spectral resampling. Feature-space alignment between source and
target domains is not performed.

---

## 4. Performance Bottlenecks

Ranked by estimated impact on final OA:

1. **Single-pass uncertainty on small labeled pool** => noisy query signal => wrong samples queried
2. **No class-aware querying** => systematic underperformance on imbalanced datasets (Pavia)
3. **Candidate pool too small (4k=80)** => farthest-first diversity operates on a restricted and potentially noisy candidate set
4. **No spatial diversity constraint** => risk of sampling spatially redundant patches
5. **Disabled dropout** => insufficient regularization on a 160-sample labeled set
6. **Naive spectral resampling for cross-sensor** => significant performance gap for AVIRIS->ROSIS

---

## 5. Active Learning Weaknesses (Deep Analysis)

### Current: LC Filter + Farthest Diversity
- LC filter selects top 4k (=80) most uncertain candidates
- Farthest-first traversal then selects k=20 diverse samples from those 80
- This is conceptually sound but the candidate pool (80) is too small

### Why It Fails on Pavia University
- Pavia University has 9 classes with very different class frequencies
- LC uncertainty concentrates on class boundaries in the feature space
- Boundary samples are harder to classify, so they dominate the LC filter
- Farthest-first diversity spreads them but still within uncertain regions
- Result: the model learns to recognize boundaries at the cost of confident majority regions
- Random avoids this by uniform sampling across the distribution

### Missing: Class-Aware Querying
No current strategy ensures that rare classes get queried. If a rare class occupies 2% of the
unlabeled pool, random selection gives it 2% representation (2xk/pool). LC or LC+FD may give
it more or less depending on the uncertainty distribution.
**No guarantee of rare-class coverage.**

### Missing: Labeled-Pool-Aware Diversity
The farthest-first traversal operates within the candidate pool (80 samples). It maximizes
diversity within those 80 samples. It does **not** ensure diversity **relative to
already-labeled data**. A sample that is diverse relative to the unlabeled pool centroid
but near an already-labeled sample is still selected.

---

## 6. Feature Extraction Weaknesses

### W-FE1: Global Average Pooling Discards Spatial Information
The feature extractor produces [B, 2048] by `global_pool(layer4_output).flatten(1)`. Layer4
has spatial resolution, and global average pooling collapses all spatial positions equally.
For a 15x15 patch, this may cause the classifier to see the same feature for patches that
differ only spatially within the patch.

### W-FE2: Only Final Layer Features Are Used
Only `layer4` output -> global pool -> [B, 2048]. Intermediate features (layer1, layer2,
layer3) are discarded. Multi-scale feature concatenation is well-established in hyperspectral
classification (especially for minority classes that have subtle spectral signatures
at different scales).

### W-FE3: SpectralAdapter Uses AdaptiveAvgPool3d
The spectral adapter pools all spectral information into a single spatial feature map at the
end. `AdaptiveAvgPool3d(1, H, W)` averages across all remaining spectral depth channels.
This is an aggressive lossy compression of the spectral axis. Spectral discrimination is
critical for fine-grained land cover classes.

### W-FE4: No Feature Caching
Every AL round re-extracts features for the entire unlabeled pool. Since the backbone is
frozen, these features are **deterministic** — they do not change between rounds.
Re-computing them every round is wasteful. With 349M FLOPs per patch and thousands of
unlabeled samples, this is a significant overhead.

---

## 7. Classifier Head Weaknesses

### W-CL1: Dropout is Disabled
As noted in W1, `Dropout(p=0.0)` means no regularization beyond weight decay. With 160
training samples and a 256-dimensional hidden layer, overfitting is likely.

### W-CL2: Kaiming Normal Initialization Assumes ReLU
`_reset_weights()` uses `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")`. The
activation is GELU. This is a minor mismatch — Kaiming for GELU is generally fine but
not theoretically optimal.

### W-CL3: No Class-Balanced Loss
`nn.CrossEntropyLoss()` with no `weight` argument treats all classes equally. With Indian
Pines having classes with as few as 20 total samples (e.g., Oats: 20, Alfalfa: 46), and
only ~7 of those in the initial seed, the loss is dominated by the large frequent classes.

### W-CL4: Temperature Scaling Is Available but Not Used
`predict_probabilities` accepts `temperature` but it is always called with `temperature=1.0`.
Post-hoc temperature scaling is one of the most effective calibration methods and could
directly improve ECE and the quality of uncertainty estimates used by Model B.

### W-CL5: Warm-Start Has Not Been Ablated
Every retrain continues from previous head weights (reset_head=False by default). This is
usually beneficial but it means the classifier accumulates biases from early rounds with
unreliable labels that came from an uncertain classifier.

---

## 8. Cross-Sensor Weaknesses

### W-CS1: Spectral Interpolation Ignores Physical Sensor Differences
Linear interpolation treats spectral bands as numerical values, ignoring:
- Gaussian spectral response functions
- Non-uniform wavelength spacing
- Out-of-range wavelength coverage (ROSIS only covers 430-860nm, missing AVIRIS 860-2500nm)

### W-CS2: FC1 Weight Transfer Is a Weak Prior
Copying FC1 weights (2048->256 projection) from source to target assumes that the
2048-dimensional feature embedding has the same semantic structure across datasets.
This may hold within AVIRIS (IP->Salinas: 74.14%) but fails cross-sensor
(IP->Pavia: 49.84%). For AVIRIS->ROSIS, the spectral adapter output at 128 channels is
produced by very different spectral content, so the 2048-dimensional embedding will
be semantically different.

### W-CS3: No Domain Alignment
The corrected transfer simply copies weights and runs target AL. There is no attempt to
align source and target feature distributions (e.g., CORAL, batch normalization alignment,
prototype matching). The backbone BatchNorm statistics were computed on training data —
they may be poorly calibrated for the target domain.

### W-CS4: Source Model Only Trains on 10 Samples/Class
The source model for transfer is trained on 10 samples/class of the source dataset. This
is a weak starting point. The FC1 weights transferred represent a very limited view of
the source feature space.

---

## 9. Evaluation Weaknesses

### W-EV1: Spatial Correlation in Train/Test Split
The 70/15/15 random pixel split means adjacent pixels can appear in train and test (15x15
patch overlap). This inflates reported OA. This is standard practice for these benchmark
datasets, but should be acknowledged in any research paper.

### W-EV2: AULC Not Computed
No Area Under the Learning Curve is reported. Final-round OA masks label efficiency.

### W-EV3: No Per-Class F1 Curve Across AL Rounds
Per-class F1 exists in the output JSON but learning curves per class across rounds are not
tracked or visualized. Minority-class recovery (e.g., does Oats get classified correctly
by round 3?) is invisible.

### W-EV4: ECE Is Measured but Not Used
ECE is computed in `trainer.py:evaluate()` but never used to improve the system (e.g., to
calibrate uncertainty before querying). Computing and discarding ECE is a lost opportunity.

### W-EV5: Cross-Sensor Transfer Runs Only 1 Seed
All cross-sensor results are single-seed (seed=42). There are no confidence intervals for
the transfer experiments. The 74.14% for IP->Salinas could be +/-3-5% across seeds.

### W-EV6: No Comparison Baseline for Transfer
There is no "within-dataset Salinas baseline" vs "IP->Salinas transfer" comparison.
Without this, it is unclear whether the FC1 transfer actually helps over pure target AL.

---

## 10. Ranked Improvement List

---

## TIER 1 — HIGH VALUE / SHOULD IMPLEMENT

These have high expected benefit, low leakage risk, and moderate implementation cost.

---

### T1-A: Enable Appropriate Dropout (p=0.1 to p=0.3)

**Current**: `Dropout(p=0.0)` — regularization completely disabled.
**Why limiting**: 160 samples training a 256-dimensional hidden layer. Overfitting is guaranteed.
**Change**: Pass `dropout=0.1` (or test 0.2) in all `ClassifierHead` instantiation calls.
**Expected effect**:
- OA: +1% to +3% (especially on Indian Pines where classes are rare)
- AA: +1% to +4% (minority class improvement)
- Label efficiency: improved (less overfitting to early queried samples)
- Cross-sensor: improved (source head generalizes better)

**Computational cost**: Zero
**Implementation**: 1 line change per script. Change `ClassifierHead(...)` to pass `dropout=0.1`
**Leakage risk**: None
**Changes architecture**: No
**Frozen backbone preserved**: Yes
**Should implement**: **YES — immediately**

---

### T1-B: Cache Frozen-Backbone Features

**Current**: Features re-extracted every AL round for the entire unlabeled pool.
**Why limiting**: Backbone is frozen — features are **deterministic**. Re-computing them is pure waste.
**Change**: Extract all features once before the AL loop starts, store in-memory as
`np.ndarray [N, 2048]`. Each round reads from cache instead of running backbone forward passes.
**Expected effect**:
- OA: No direct effect on quality (features are identical)
- Speed: 5-10x faster per AL round for feature extraction step
- Label efficiency: indirect benefit — faster iteration means more experiments possible

**Computational cost**: Saves cost
**Implementation**: ~30 lines — extract once, cache, load in each round
**Leakage risk**: None (unlabeled pool features do not include labels)
**Changes architecture**: No (only the training loop)
**Frozen backbone preserved**: Yes
**Should implement**: **YES — immediately**

---

### T1-C: Labeled-Pool-Aware Diversity

**Current**: Diversity scores measure distance from the **unlabeled pool centroid**.
Farthest-first in `select_diverse_samples()` maximizes intra-candidate diversity.
**Why limiting**: Neither considers what is **already labeled**. Samples can be diverse
within themselves but near existing labeled data.
**Change**: In `filter_diversity`, after selecting the top-k LC candidates, run farthest-first
with respect to **labeled pool embeddings** as the initial "selected set" (i.e., start the
greedy traversal from the existing labeled embeddings). This ensures selected samples are
far from **already-labeled** data, not just from each other.
**Expected effect**:
- OA: +1% to +3% (better coverage of unlabeled space)
- AA: +2% to +4% (especially for minority classes that are far from initial seed)
- Label efficiency: improved across all datasets

**Computational cost**: Requires storing labeled pool embeddings (~160 x 2048 floats = negligible with caching)
**Implementation**: Modify `select_diverse_samples()` to accept an optional `existing_embeddings`
parameter that seeds the initial distance matrix
**Leakage risk**: None (labeled pool labels are already known; embeddings are fine)
**Changes architecture**: No
**Frozen backbone preserved**: Yes
**Should implement**: **YES**

---

### T1-D: Class-Balanced Cross-Entropy Loss

**Current**: `nn.CrossEntropyLoss()` with equal weight for all classes.
**Why limiting**: Indian Pines has classes with 20-2400 total pixels. Majority classes dominate the loss.
**Change**: Compute inverse class frequency weights from the labeled pool at each round.
Pass as `weight=` to `CrossEntropyLoss`. Weights should be recomputed each round as
new samples are added.
**Expected effect**:
- OA: May decrease slightly (as model better handles minorities)
- AA: +3% to +7% on Indian Pines (direct minority class improvement)
- Macro-F1: +2% to +5%
- Kappa: +2% to +4%
- Pavia University OA gap vs Random: may reduce

**Computational cost**: Zero (weight computation is O(C) per round)
**Implementation**: ~10 lines in `trainer.py`
**Leakage risk**: None (class frequencies computed from **labeled pool only**)
**Changes architecture**: No
**Frozen backbone preserved**: Yes
**Should implement**: **YES**

---

### T1-E: Post-Hoc Temperature Scaling for Better-Calibrated Uncertainty

**Current**: Uncertainty computed from raw softmax probabilities (ECE can be significant).
Temperature is hardcoded to 1.0.
**Why limiting**: Poorly calibrated probabilities -> LC scores don't reflect true uncertainty
-> wrong samples queried.
**Change**: After each classifier training round, perform temperature scaling calibration
using the **validation set** (not test set). Find optimal T via 1D search (bisection on NLL).
Use scaled probabilities in the query engine.
**Expected effect**:
- OA: +1% to +2% (better AL query signal)
- AA: +1% to +3%
- ECE: significant reduction
- Cross-sensor: improved (calibrated source probabilities better guide transfer)

**Computational cost**: Negligible (1D optimization over val set)
**Implementation**: ~25 lines
**Leakage risk**: Uses validation set (not test set) — **zero leakage**
**Changes architecture**: No
**Frozen backbone preserved**: Yes
**Should implement**: **YES**

---

### T1-F: AULC Computation and Reporting

**Current**: Only final-round OA/F1 are compared across strategies.
**Why limiting**: Does not capture label efficiency. A strategy that reaches 65% at round 2
is better than one reaching 65% at round 5 with the same budget.
**Change**: Compute `AULC = sum(OA_r) / (total_rounds + 1)` across all rounds (including
round 0). Report and compare strategies on AULC, not just final OA.
**Expected effect**: No model quality change, but provides scientifically stronger claim
for label efficiency advantage of LC+FD.

**Computational cost**: Zero (data already logged)
**Implementation**: 5 lines in result processing
**Leakage risk**: None
**Should implement**: **YES — immediately, before any paper submission**

---

## TIER 2 — POTENTIALLY HIGH VALUE / TEST FIRST

These are promising but require controlled experiments to confirm benefit.

---

### T2-A: Proportional Candidate Pool in filter_diversity

**Current**: `cand_pool_size = min(len(embeddings), max(4*k, 50))` => always 80 with k=20.
**Why potentially valuable**: 80 out of 7000+ unlabeled samples (1.1%) is very restrictive.
The LC ranking is unreliable early in training. Using top-20% (1400 samples) would give
farthest-first diversity a much richer space to explore, reducing the chance of selecting
noisy boundary artifacts.
**Proposed change**: `cand_pool_size = min(len(embeddings), max(int(0.20 * len(embeddings)), 4 * k))`
**Expected effect**:
- OA: +1% to +3% on Pavia University (where current method underperforms random)
- OA: neutral to +1% on Indian Pines, Salinas
- Risk: if classifier is poorly calibrated early, a larger candidate pool could include more noise

**Recommendation**: Run a controlled 5-seed comparison with candidate pool = {4k, 10%, 20%, 50%}
on all three datasets. The result will determine the optimal pool fraction.
**Leakage risk**: None
**Implementation**: 1 line change

---

### T2-B: MC Dropout Uncertainty (Multiple Stochastic Forward Passes)

**Current**: Single deterministic forward pass uncertainty (LC, Entropy, Margin).
**Why potentially valuable**: A model trained on 160 samples with one forward pass produces
overconfident softmax outputs. MC Dropout with T=10-20 forward passes with `Dropout(p>0)`
active would give approximate Bayesian uncertainty estimates. The variance in predictions
across passes is a better proxy for model uncertainty than max-softmax.
**Requirement**: Dropout must first be enabled (T1-A). Only meaningful if dropout > 0.
**Implementation**: Enable dropout, run T forward passes in eval mode with dropout active,
compute predictive entropy of mean probabilities. ~20 lines.
**Expected effect**:
- Query quality improvement: moderate (+1-3% OA)
- Risk: T=20 passes multiplies feature extraction time 20x unless features are cached (T1-B)

**Note**: With feature caching, the overhead is only T classifier forward passes on
7000 x 2048 embeddings, which is very fast (~0.01 sec).
**Leakage risk**: None
**Should test before implementing at scale**: YES

---

### T2-C: Class-Aware Query Constraint (Minimum Class Representation)

**Current**: No guarantee that rare classes are queried.
**Proposed**: After standard selection, check which classes are under-represented in the
labeled pool. If any class has fewer than `min_per_class` labeled samples, force-sample
at least 1 sample from that class's unlabeled pool (by choosing the most uncertain
sample in that predicted class).
**Expected effect**:
- AA: +3% to +6% on Indian Pines (directly addresses rare class starvation)
- OA: likely slight decrease or neutral (rare class samples are a minority of queried budget)
- Macro-F1: +3% to +5%

**LEAKAGE WARNING**: Requires knowing which class an unlabeled sample belongs to, which
breaks the active learning assumption. A safe workaround: track the **predicted** class
distribution from Model B and ensure query diversity across predicted classes, not true
labels. Never use true labels of unlabeled samples.
**Implementation**: Moderate (~50 lines)
**Should test carefully**: YES, but only using predicted labels, never true labels

---

### T2-D: Feature Embedding Normalization Before Classifier Input

**Current**: `input_norm = LayerNorm(2048)` normalizes per-sample across all 2048 dimensions.
**Alternative**: L2-normalize embeddings before the classifier head. This projects all
embeddings onto a unit hypersphere, which can improve linear separability and reduce
scale variance between datasets.
**Expected effect**:
- OA: +1-2% on cross-sensor (features from different sensors have different L2 norms)
- Within-dataset: neutral to +1%

**Note**: LayerNorm is already close to this. L2 normalization is a simpler and stronger constraint.
**Implementation**: Replace `LayerNorm(2048)` input_norm with `F.normalize(features, dim=-1)` before the head.
**Risk**: May hurt performance if class distances rely on magnitude differences.

---

### T2-E: Multi-Scale Feature Fusion (Layer3 + Layer4)

**Current**: Only `layer4` global pool -> [B, 2048].
**Proposed**: Concatenate `layer3` global pool ([B, 1024]) + `layer4` global pool ([B, 2048])
-> [B, 3072], then feed into classifier head.
**Expected effect**:
- OA: +1-3% on minority classes (layer3 features may capture finer spectral textures)
- AA: +2-4%

**Computational cost**: layer3 features are already computed during backbone forward pass —
no extra backbone compute. Only the classifier input dimension changes to 3072.
**Implementation**: Modify `feature_extractor.py` to return both layer3 and layer4 features.
Change classifier head `in_features` to 3072.
**Risk**: Increases trainable parameters from 533K to ~790K. More parameters may hurt with
small labeled pools unless dropout is enabled.
**Should test with dropout enabled**: YES, in combination with T1-A

---

### T2-F: Cross-Sensor Transfer: CORAL Feature Alignment

**Current**: Raw FC1 weights transferred, no domain alignment.
**Proposed**: Before target AL, align the second-order statistics (covariance) of source
and target embeddings. This is CORAL (Correlation Alignment), a simple and computationally
cheap domain adaptation method.

**Steps**:
1. Extract embeddings for all target unlabeled samples.
2. Extract embeddings for all source labeled samples.
3. Whiten source embeddings (subtract mean, normalize covariance).
4. Apply source covariance structure to target embeddings.
5. Proceed with target AL on covariance-aligned embeddings.

**Note**: The backbone is frozen, so feature alignment happens post-backbone.
No backbone modification needed.
**Expected effect**:
- Cross-sensor OA: +3-8% (especially AVIRIS->ROSIS transfer)

**Implementation**: ~30 lines
**Leakage risk**: None (unlabeled target embeddings have no labels)

---

## TIER 3 — LOW VALUE / NOT WORTH THE TIME

These ideas are either unlikely to improve actual metrics or add unjustified complexity.

---

### T3-A: BADGE (Batch Active Learning by Diverse Gradient Embeddings)

**Why not valuable here**: BADGE computes gradient embeddings of the output layer's parameters
for each unlabeled sample. With a frozen backbone and a 2-layer MLP head, gradient embeddings
collapse to the outer product of the final layer gradient and the penultimate activation.
This is essentially a diversity measure in gradient space. In practice for this setup, it
will perform similarly to farthest-first diversity in embedding space (T1-C), at
significantly higher computational cost.

### T3-B: Pseudo-Labeling or Self-Training

Requires unfreezing or modifying the backbone, or using noisy pseudo-labels from an
unreliable small-pool classifier. High leakage risk. Explicitly excluded from project scope.
**Do not implement.**

### T3-C: TypiClust

Useful for the very first round (selecting the initial seed). Since the initial seed is
already class-balanced stratified, TypiClust provides marginal benefit.
After round 0, uncertainty-based methods are more appropriate.

### T3-D: Larger Hidden Dimension (256 -> 512)

The bottleneck is the labeled pool size (160 samples), not the classifier capacity.
Doubling hidden dimension increases parameters from 533K to ~1.1M without improving
generalization on 160 samples. Wait until the labeled pool is larger or enable
proper dropout first.

### T3-E: Superpixel-Based Query Selection

Requires spatial coordinate information from the dataset manager (currently not passed
to Model B). Complex to implement. Limited benefit unless spatial redundancy is very severe.

### T3-F: BALD (Bayesian Active Learning by Disagreement)

Theoretically appealing. Requires MC Dropout (T2-B) as prerequisite. However, with 5 rounds
and k=20, BALD's advantage over simpler methods is typically small in low-budget AL regimes.

### T3-G: Cosine Classifier

Cosine classifiers help in few-shot learning settings but their benefit over a standard
linear head + LayerNorm is marginal when the feature extractor is a strong pretrained model.
Not worth testing given current priorities.

---

## 11. Expected Benefit Table

| ID | Improvement | OA Delta | AA Delta | Macro-F1 Delta | Label Efficiency | Cross-Sensor | Compute Cost | Risk |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1-A** | Enable Dropout (0.1) | +1-3% | +2-4% | +2-4% | Up | Up | Zero | Very Low |
| **T1-B** | Cache Backbone Features | 0% | 0% | 0% | Faster | None | Saves | None |
| **T1-C** | Labeled-Pool-Aware Diversity | +1-3% | +2-4% | +2-3% | Up | Up | Very Low | None |
| **T1-D** | Class-Balanced Loss | -1-0% | +3-7% | +2-5% | Up minority | None | Zero | None |
| **T1-E** | Temperature Scaling | +1-2% | +1-3% | +1-2% | Up | Up | Negligible | None |
| **T1-F** | AULC Reporting | 0 | 0 | 0 | Visible | None | Zero | None |
| T2-A | Proportional Candidate Pool | 0-3% | 0-2% | 0-2% | Up | None | Zero | Low |
| T2-B | MC Dropout Uncertainty | +1-3% | +1-3% | +1-2% | Up | Up | Low (w/cache) | Low |
| T2-C | Class-Aware Query Constraint | -1-0% | +3-6% | +2-5% | Up minority | None | Low | Medium |
| T2-D | L2 Feature Normalization | 0-2% | 0-2% | 0-1% | None | +2-5% | Zero | Low |
| T2-E | Multi-Scale Feature Fusion | +1-3% | +2-4% | +1-3% | None | None | Low | Low |
| T2-F | CORAL Feature Alignment | None | None | None | None | +3-8% | Low | Low |
| T3-A | BADGE | 0-1% | 0-1% | 0-1% | None | None | High | Low |
| T3-E | Superpixels | 0-2% | 0-2% | 0-1% | Up | None | High | Medium |

---

## 12. Computational Cost Summary

| Improvement | Additional Compute per AL Run |
| :--- | :--- |
| T1-A (Dropout) | Zero |
| T1-B (Feature Cache) | -80% of feature extraction time |
| T1-C (Labeled-Pool Diversity) | O(n_labeled x D) — negligible |
| T1-D (Class-Balanced Loss) | O(n_labeled) per round — negligible |
| T1-E (Temperature Scaling) | 1D bisection over val set — negligible |
| T2-A (Proportional Pool) | O(n_unlabeled log n) for sorting — already done |
| T2-B (MC Dropout, T=20) | 20x classifier forward on unlabeled (fast with cache) |
| T2-E (Multi-Scale) | Slight increase in classifier forward pass |
| T2-F (CORAL) | O(D^2 x n_source) covariance computation |

---

## 13. Implementation Complexity

| Improvement | Lines Changed | Files Changed | Complexity |
| :--- | :---: | :---: | :--- |
| T1-A | ~6 | 2 scripts | **Trivial** |
| T1-B | ~40 | learning_engine.py or scripts | **Low** |
| T1-C | ~20 | diversity.py, engine.py | **Low** |
| T1-D | ~15 | trainer.py | **Low** |
| T1-E | ~30 | trainer.py or learning_engine.py | **Low** |
| T1-F | ~15 | run_multiseed_validation.py | **Trivial** |
| T2-A | ~2 | engine.py | **Trivial** |
| T2-B | ~30 | engine.py + trainer.py | **Medium** |
| T2-C | ~60 | engine.py + dataset_manager.py | **Medium** |
| T2-E | ~40 | feature_extractor.py | **Medium** |
| T2-F | ~40 | run_cross_sensor_eval.py | **Low** |

---

## 14. Research Novelty Potential

| Improvement | Novelty for Hyperspectral AL |
| :--- | :--- |
| T1-C (Labeled-Pool Diversity) | Moderate — rarely analyzed separately in HSI AL papers |
| T1-D + T2-C (Class-Balanced + Constrained Query) | High — class imbalance in HSI AL is underexplored |
| T1-E (Calibration-Aware AL) | Moderate — ECE-aware querying is novel for HSI |
| T2-B (MC Dropout AL) | Low — well-studied in general AL |
| T2-F (CORAL for cross-sensor) | High — cross-sensor domain alignment in few-shot AL is novel |
| T2-E (Multi-scale features from frozen backbone) | Moderate — simple but empirically validated |

---

## 15. Recommended Experiment Order

**Phase 1 — Zero-Cost Fixes (implement before any new experiment)**
1. `T1-A`: Enable dropout=0.1 everywhere
2. `T1-F`: Add AULC computation to multiseed results
3. `T2-A`: Change candidate pool to proportional (20%)

**Phase 2 — High-Value Changes (run 5-seed validation after each)**
4. `T1-B`: Implement feature caching (enables faster iteration for all remaining experiments)
5. `T1-D`: Add class-balanced loss to trainer
6. `T1-C`: Implement labeled-pool-aware diversity
7. `T1-E`: Add temperature scaling calibration

**Phase 3 — Experimental Extensions (test carefully, compare to Phase 2 baseline)**
8. `T2-B`: MC Dropout uncertainty (requires T1-A + T1-B)
9. `T2-E`: Multi-scale feature fusion (test impact on minority classes)
10. `T2-F`: CORAL feature alignment for cross-sensor

**Phase 4 — Optional Research Extensions**
11. `T2-C`: Class-constrained query (predicted class only, no leakage)
12. `T2-D`: L2 feature normalization ablation

---

## 16. Final Recommended Architecture

After implementing Tier 1 improvements, the final system should be:

```
Input patch [B, C, 15, 15]
      |
SpectralAdapter (frozen, pretrained)   -- unchanged
      |
CustomResNet50 (frozen, pretrained)    -- unchanged
  global_pool -> [B, 2048]             -- CACHE this output
      |
      | (CACHED -- extracted once before AL loop)
      |
ClassifierHead [MODIFIED]:
  LayerNorm(2048)
  Linear(2048 -> 256)
  LayerNorm(256)
  GELU()
  Dropout(p=0.1)                      <- T1-A: enabled
  Linear(256 -> C)
      |
CrossEntropyLoss(weight=class_weights) <- T1-D: class-balanced
      |
After training: Temperature Scaling    <- T1-E: calibrated probabilities
      |
QueryEngine [MODIFIED]:
  LC -> proportional candidate pool    <- T2-A: 20% of unlabeled
  Farthest-first with labeled pool    <- T1-C: labeled-pool-aware diversity
  -> k selected indices
```

---

## 17. What NOT to Change

| Component | Reason to Leave Alone |
| :--- | :--- |
| **Backbone freezing mechanism** | Correctly implemented. Strict eval mode even during classifier training. |
| **AdamW + CosineAnnealingLR** | Well-tuned for the small labeled pool. No evidence it limits performance. |
| **Farthest-first diversity kernel** | Correct implementation of greedy k-center. Only the input diversity set needs modification (T1-C). |
| **Class-balanced initial seed** | Already class-stratified at 10/class. Correct. |
| **5 seeds x 3 strategies x 3 datasets validation** | Sufficient for statistical rigor. Do not reduce. |
| **Dataset splits (70/15/15 random pixel)** | Standard benchmark practice. Changing would invalidate comparison with published results. |
| **Model A / Model B separation** | Clean, correct, and architecturally sound. |
| **Gradient clipping (max_norm=1.0)** | Correct for small labeled pool training. |
| **LC Filter + Farthest Diversity as primary strategy** | Within-dataset results confirm it is the strongest strategy on 2 of 3 datasets. Keep as default. |
| **k=20, 5 rounds, patch_size=15** | Standard budget. Do not change without a separate ablation study. |

---

## 18. Final Recommendation

**The core architecture is sound.** The SpectralEarth backbone is a strong feature extractor,
the farthest-first diversity selection is theoretically correct, and the experimental
methodology is rigorous.

**The three highest-leverage improvements are**:

1. **Enable Dropout (T1-A)**: This is the single most critical omission. A classifier head
   with 533K parameters being trained on 160 samples with `p=0.0` dropout is near-certain
   to overfit. Enable `p=0.1`. This is a one-line change per script and costs nothing.

2. **Labeled-Pool-Aware Diversity (T1-C)**: The current farthest-first traversal selects
   samples that are diverse relative to each other within the candidate pool, but ignores
   what is already labeled. This is a fundamental limitation of the current diversity
   criterion. Making it labeled-pool-aware is the most principled improvement to Model B.

3. **Class-Balanced Loss (T1-D)**: The Pavia University results (random beating active
   learning) and the generally low AA and Macro-F1 vs OA across all datasets are diagnostic
   of class imbalance. Inverse-frequency weighting is a well-established and zero-cost fix.

**The most dangerous antipattern in this codebase** is the `dropout=0.0` default that silently
disables regularization. Without discovering this through code inspection, all experiments have
been run with a systematically under-regularized classifier head.

**Do not unfreeze the backbone. Do not replace the architecture. Do not add pseudo-labeling.**
The gains from the three Tier 1 modifications above should be sufficient to meaningfully
improve OA, AA, and Macro-F1 across all datasets, with particular benefit for Indian Pines
minority classes and Pavia University's active learning gap against random.

---

*End of Independent Audit Report*