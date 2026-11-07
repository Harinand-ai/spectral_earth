# Extended Label-Budget Accuracy Study: Experimental Methodology

## 1. Experimental Architecture and Controls
This study investigates whether expanding the labeled training budget from $N=220$ up to $N=2000$ allows the best-performing active learning strategies to achieve high-accuracy regimes ($90\%$, $95\%$, or $99\%$).

### 1.1 Frozen Foundation Model Backbone
- **Model:** SpectralEarth 3D-ResNet50 pretrained on multi-sensor hyperspectral imagery.
- **Parameters:** Completely frozen (`requires_grad=False`, $\nabla_\theta = 0$).
- **Features:** Unit $L_2$-normalized 2048-dimensional embeddings ($\|z\|_2 = 1.0$).
- **Cache Isolation:** Pre-extracted embeddings loaded directly from disk cache; no feature re-computation or backbone fine-tuning.

### 1.2 Classifier Architecture (Head B - Improved 512)
- **Topology:** $\text{Linear}(2048 \to 512) \to \text{BatchNorm1d}(512) \to \text{GELU}() \to \text{Dropout}(0.10) \to \text{Linear}(512 \to C)$.
- **Parameter Count:** 1,058,320 (~1.06M trainable parameters).
- **Optimizer:** AdamW (initial learning rate $\eta = 0.002$, weight decay $\lambda = 0.001$).
- **Learning Rate Schedule:** CosineAnnealingLR over $T_{\text{max}} = 45$ epochs.
- **Batch Size:** 64 with drop-last guard on single-sample residual batches.

### 1.3 Dataset Splitting and Leakage Safeguards
- **Stratified Partition:** 10% candidate training pool, 10% validation pool, 80% strictly held-out test pool.
- **Available Training Pools:**
  - Indian Pines: 7,174 pixels
  - Salinas: 37,890 pixels
  - Pavia University: 29,943 pixels
- All target budgets ($N \in \{220, 300, 400, 500, 700, 1000, 1500, 2000\}$) are strictly within the available training pool size.
- **Oracle Labeling:** Test pixels are strictly prohibited from entering training, validation, or acquisition scoring.

## 2. Selected Active Learning Acquisition Strategies
Based on the validated benchmark findings, the top acquisition strategy was selected independently for each scene:
1. **Indian Pines:** Margin + Cosine Diversity (`hybrid_margin_cosine`)
2. **Salinas:** Margin + Cosine Diversity (`hybrid_margin_cosine`)
3. **Pavia University:** Margin Sampling (`margin`)

### Phase Sequence:
- **Phase A (Active Learning):** Active learning acquisition across budgets $N \in \{220, 300, 400, 500, 700, 1000, 1500, 2000\}$ across 5 random seeds (42–46). Cumulative acquisition policy ($L_{t-1} \subset L_t$).
- **Phase B (Random Sampling):** Random pixel selection from the identical training pool at the identical budgets and seeds.
