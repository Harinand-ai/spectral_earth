# ConformAL-HSI: Final Experimental Methodology & Protocol

**Framework:** ConformAL-HSI (Conformalized Active Learning for Hyperspectral Image Classification)  
**Foundation Model:** SpectralEarth Pretrained 3D-ResNet50 Backbone (Frozen, $\nabla_\theta = 0$)  
**Feature Dimension:** $d = 2048$ ($L_2$-normalized: $\|f(x)\|_2 = 1$)  
**Optimization Environment:** PyTorch 2.x, CUDA 12.x, NVIDIA GeForce RTX 2050 (4GB VRAM)  
**Datasets:** Indian Pines ($145 \times 145$, 16 classes), Salinas ($512 \times 217$, 16 classes), Pavia University ($610 \times 340$, 9 classes)  

---

## 1. Classifier Head Architectures

To definitively isolate whether classifier capacity, architectural depth, or normalization alters the effectiveness and ranking of Active Learning acquisition strategies, we benchmark four precisely specified classifier heads:

### 1.1 Head A — Original Baseline (256 units)
- **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{2048} \xrightarrow{\text{Linear}(2048, 256)} \mathbf{h} \in \mathbb{R}^{256} \xrightarrow{\text{ReLU}} \mathbf{a} \in \mathbb{R}^{256} \xrightarrow{\text{Linear}(256, C)} \mathbf{z} \in \mathbb{R}^{C}$$
- **Regularization:** None (no normalization, no dropout).
- **Parameters:** $528,656$ parameters (Indian Pines / Salinas: $C=16$), $526,857$ parameters (Pavia University: $C=9$).

### 1.2 Head B — Improved Architecture (512 units)
- **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{2048} \xrightarrow{\text{Linear}(2048, 512)} \mathbf{h} \xrightarrow{\text{BatchNorm1d}(512)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.1)} \xrightarrow{\text{Linear}(512, C)} \mathbf{z} \in \mathbb{R}^{C}$$
- **Regularization:** Mini-batch BatchNorm1d ($\epsilon=1\text{e-}5, \text{momentum}=0.1$), Dropout ($p=0.10$).
- **Parameters:** $1,058,320$ parameters ($C=16$), $1,054,729$ parameters ($C=9$).

### 1.3 Head C — Wider Architecture (1024 units)
- **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{2048} \xrightarrow{\text{Linear}(2048, 1024)} \mathbf{h} \xrightarrow{\text{BatchNorm1d}(1024)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.2)} \xrightarrow{\text{Linear}(1024, C)} \mathbf{z} \in \mathbb{R}^{C}$$
- **Regularization:** Mini-batch BatchNorm1d, Dropout ($p=0.20$).
- **Parameters:** $2,116,624$ parameters ($C=16$), $2,109,449$ parameters ($C=9$).

### 1.4 Head D — Deeper Multi-Stage Architecture (1024-512-256 units)
- **Architecture:**
  $$\mathbf{x} \in \mathbb{R}^{2048} \xrightarrow{\text{Linear}(2048, 1024)} \xrightarrow{\text{BN}(1024)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.20)}$$
  $$\xrightarrow{\text{Linear}(1024, 512)} \xrightarrow{\text{BN}(512)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.10)}$$
  $$\xrightarrow{\text{Linear}(512, 256)} \xrightarrow{\text{GELU}} \xrightarrow{\text{Linear}(256, C)} \mathbf{z} \in \mathbb{R}^{C}$$
- **Regularization:** Two-stage BatchNorm1d and staggered Dropout ($p_1=0.20, p_2=0.10$).
- **Parameters:** $2,761,488$ parameters ($C=16$), $2,759,689$ parameters ($C=9$).

---

## 2. Normalization Diagnostic Formulations

To evaluate the residual mini-batch hypothesis at budget $N=260$ ($260 = 4 \times 64 + 4$ samples):
1. **BatchNorm1d:** Normalizes across the batch dimension:
   $$\hat{x}_{bi} = \frac{x_{bi} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad \mu_i = \frac{1}{B}\sum_{b=1}^B x_{bi}, \quad \sigma_i^2 = \frac{1}{B}\sum_{b=1}^B (x_{bi} - \mu_i)^2$$
   *Vulnerability:* When $B=4$, sample variance $\sigma_i^2$ exhibits high estimation error, corrupting running statistics.
2. **LayerNorm:** Normalizes across feature dimensions per sample independently of batch size:
   $$\hat{x}_{bi} = \frac{x_{bi} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}, \quad \mu_b = \frac{1}{D}\sum_{j=1}^D x_{bj}, \quad \sigma_b^2 = \frac{1}{D}\sum_{j=1}^D (x_{bj} - \mu_b)^2$$
   *Advantage:* Invariant to mini-batch size $B$, ensuring identical statistics for full batches ($B=64$) and residual batches ($B=4$).

---

## 3. Active Learning Acquisition Formulations

1. **Random Sampling:**
   $$i^* \sim \text{Uniform}(U_t)$$
2. **Margin Sampling:**
   $$i^* = \arg\min_{i \in U_t} \left( p_i^{(1)} - p_i^{(2)} \right)$$
   where $p_i^{(1)}$ and $p_i^{(2)}$ are the largest and second largest predicted class probabilities for unlabeled sample $i$.
3. **Margin + Cosine Diversity:**
   $$\text{Score}(i) = \alpha \cdot \left( 1 - \left( p_i^{(1)} - p_i^{(2)} \right) \right) + (1 - \alpha) \cdot \min_{j \in L_t} \left( 1 - \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\|\mathbf{f}_i\|_2 \|\mathbf{f}_j\|_2} \right)$$
   with $\alpha = 0.60$.
4. **Adaptive Class-Coverage AL:**
   $$\text{CoverageScore}(i) = \left( 1 - \left( p_i^{(1)} - p_i^{(2)} \right) \right) \times \left( \frac{\max_k N_k}{\max(1, N_{\hat{y}_i})} \right)^\gamma$$
   with balancing exponent $\gamma = 0.50$, where $N_k$ is the current count of labeled samples in class $k$ and $\hat{y}_i = \arg\max_k p_{ik}$.

---

## 4. Controlled Training & Evaluation Protocol

- **Epochs:** 40 epochs per acquisition round.
- **Batch Size:** 64.
- **Optimizer:** AdamW ($\text{lr} = 1\text{e-}3$, weight decay $= 1\text{e-}4$).
- **Learning Rate Schedule:** Cosine Annealing with $\eta_{\min} = 1\text{e-}5$.
- **Validation Split:** 10% strictly held-out from training and acquisition.
- **Test Split:** 80% strictly held-out until model evaluation.
- **Temperature Scaling:** Post-hoc calibration on validation logits using L-BFGS to compute optimal scaling parameter $T^* > 0$.

