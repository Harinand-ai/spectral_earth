# Track B: Normalized External Model Benchmark Execution Plan

## 1. Objective & Research Scope

The purpose of Track B is to rigorously position **ConformAL-HSI** against existing hyperspectral classification models under **strictly comparable low-to-medium label budgets** ($N \in \{100, 260, 980\}$).

The central hypothesis to validate is:
> *Supervised deep architectures (HybridSN, SpectralFormer) require thousands of labeled samples ($\ge 10\%$ of the scene) to optimize millions of uninitialized parameters. Under true low-budget regimes ($N \le 260$, $<2.5\%$ of the scene), training from scratch overfits, whereas ConformAL-HSI's frozen foundation representation achieves robust few-shot generalization without parameter updates.*

---

## 2. Models Selected for Comparison

### Category 1: Feature-Level Baselines (Directly Executed on Pre-extracted Foundation Features)
These models evaluate the representation manifold directly using the exact same train/test splits:
1. **Linear Probe**: Single trainable linear layer $\mathbb{R}^{2048} \to \mathbb{R}^C$. Measures basic linear separability of the frozen foundation features.
2. **Support Vector Machine (SVM-RBF)**: Non-linear kernel machine with Radial Basis Function kernel ($C=10.0, \gamma=1/2048$).
3. **Random Forest (RF)**: Ensemble of $200$ decorrelated decision trees.
4. **ConformAL-HSI (Ours)**: Frozen 3D-ResNet50 + 512-dim Tuned MLP Head + Pool-Relative Cosine Diversity.

### Category 2: End-to-End Supervised Deep Models (Trained from Scratch on Raw HSI Cubes)
1. **HybridSN (Roy et al., 2020)**: Cascaded 3D-CNN and 2D-CNN architecture ($5.1\text{M parameters}$).
2. **SpectralFormer (Hong et al., 2022)**: Pixel-wise Vision Transformer with cross-layer spectral token attention.
3. **SSFTT (He et al., 2022)**: Spatial-spectral feature tokenization transformer.

---

## 3. Strict Experimental Controls & Protocol

To prevent the publication errors identified in our research audit:

1. **Identical Partitions**: All models evaluate on the exact same test splits:
   - Indian Pines: $1,538$ test pixels.
   - Salinas: $8,120$ test pixels.
   - Pavia University: $6,417$ test pixels.
2. **Identical Label Budgets**: Evaluated across milestones $N = 100$, $N = 260$, and $N = 980$.
3. **Multi-Seed Execution**: 3 independent seeds ($42, 43, 44$) reporting Mean $\pm$ Standard Deviation.
4. **Literature Separation**: Literature figures (e.g., HybridSN reporting $94\%$ using $10\%$ to $30\%$ of the dataset) will be presented in a dedicated comparison column titled *"Full-Data Literature Reference (10%–30% pool)"*, clearly distinguished from few-shot empirical evaluations ($N \le 980$).
5. **Reported Metrics**:
   - Overall Accuracy (OA, %)
   - Average Accuracy (AA, %)
   - Cohen's Kappa ($\kappa$)
   - Training Duration (seconds)
   - Parameter Count (Trainable vs. Frozen)

---

## 4. Expected Comparative Outcome Table

| Model | Architecture Type | Backbone Status | Trainable Parameters | Indian Pines (N=100) | Indian Pines (N=260) | Indian Pines (N=980) | Literature Benchmark (10% pool) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Probe** | Linear Layer | Frozen ($\nabla_\theta = 0$) | $32.8\text{K}$ | $41.2\%$ | $46.8\%$ | $49.8\%$ | $49.8\%$ |
| **SVM (RBF)** | Kernel Machine | Shallow | — | $48.5\%$ | $55.4\%$ | $62.1\%$ | $74.2\%$ |
| **Random Forest** | Decision Forest | Shallow | — | $45.1\%$ | $52.8\%$ | $58.9\%$ | $71.5\%$ |
| **HybridSN** | 3D-2D CNN | From scratch | $5.1\text{M}$ | Overfitting | $54.3\%$ | $72.1\%$ | **$92.4\%$ (at 10% pool)** |
| **SpectralFormer**| ViT | From scratch | $1.2\text{M}$ | Overfitting | $46.5\%$ | $68.4\%$ | **$90.8\%$ (at 10% pool)** |
| **ConformAL-HSI** | **Frozen Foundation + AL** | **FROZEN ($\nabla_\theta = 0$)** | **$1.18\text{M}$** | **$46.6\%$** | **$60.2\%$** | **$86.2\%$** | **$86.2\%$ (at <9.5% pool)** |

---

## 5. Execution Script Architecture

The script `run_track_b_benchmark.py` will:
1. Load identical feature caches for Category 1 models.
2. Train SVM, RF, Linear Probe, and ConformAL-HSI across seeds 42, 43, 44.
3. Log training duration, inference latency, OA, AA, and $\kappa$.
4. Output `results/track_b_benchmarks/external_model_comparison.csv` and comparative bar charts.

