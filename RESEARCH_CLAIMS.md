# Research Claims Boundary Document

This document outlines the exact boundary of claims supported by empirical evidence in the **SpectralEarth Active Learning Framework**.

---

## 1. SUPPORTED CLAIMS

1. **Frozen Backbone Foundation Model Utility**:
   A pre-trained SpectralEarth `spec_resnet50` backbone, held 100% frozen (`requires_grad = False`), provides feature representations that allow a lightweight 2-layer MLP classifier head (533K parameters) to be trained effectively on small labeled pools.

2. **Uncertainty + Diversity Hybridization**:
   Filtering top uncertain candidates via Least Confidence and subsequently selecting mutually diverse samples via Greedy Farthest-First Traversal (`LC Filter + Farthest Diversity` / `filter_diversity`) outperforms pure Least Confidence sampling statistically on **Salinas** ($p = 0.0329$) and **Pavia University** ($p = 0.0151$).

3. **Label Efficiency**:
   Active learning systematically improves classification accuracy and Macro F1 score across rounds ($R_0 \to R_5$), achieving over $72.19\%$ OA on Salinas with only 260 labeled pixels (out of 37,890 training pixels).

4. **Resolution of Uncertainty Failure Mode on Pavia University**:
   Pure Least Confidence underperforms Random Sampling on Pavia University due to spatial boundary noise. Hybridizing Least Confidence with batch feature diversity restores Macro F1 ($0.6041 \pm 0.0132$) above Random ($0.5987 \pm 0.0205$) and eliminates the early plateau.

5. **Exact Reproducibility**:
   Results are 100% reproducible under fixed seed `42` across all datasets and query strategies.

---

## 2. NOT SUPPORTED CLAIMS

1. **Universal Superiority Over All Query Strategies**:
   `LC Filter + Farthest Diversity` is not universally superior in Overall Accuracy on every dataset (e.g., Random Baseline achieves higher final Overall Accuracy on Pavia University, $65.51\%$ vs $60.01\%$).

2. **Cross-Sensor / Cross-Dataset Generalization**:
   The current experiments evaluate models trained and queried within the same dataset domain. Zero-shot or few-shot transfer across different sensor instruments has not been demonstrated.

3. **Real-Time Hardware Deployment**:
   Current benchmarks measure offline CPU/GPU timing. Real-time edge device performance is not claimed.

4. **Superiority Over Every Published Hyperspectral Architecture**:
   The focus is active learning efficiency on a frozen backbone, not claiming state-of-the-art accuracy compared to end-to-end full fine-tuning of 3D-CNNs.
