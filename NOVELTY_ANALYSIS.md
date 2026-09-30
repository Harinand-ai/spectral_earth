# NOVELTY ANALYSIS & LITERATURE COMPARISON — SPECTRALEARTH AL

## Overview
This document evaluates the methodological position of the **SpectralEarth Active Learning Framework** relative to existing literature in hyperspectral image (HSI) classification, active learning, domain adaptation, and model calibration.

---

## 1. Literature Background & Context

Active learning (AL) for hyperspectral image classification has been widely studied:
- **Uncertainty Sampling**: Methods such as Least Confidence (Settles, 2009), Entropy (Dagan & Engelson, 1995), and Margin Sampling (Scheffer et al., 2001) are standard baselines in HSI active learning (Tuia et al., 2011; Li et al., 2013).
- **Diversity & Batch Active Learning**: Greedy farthest-first traversal (Hochbaum & Shmoys, 1985) and k-means center selection (Sener & Savarese, 2018 - Coreset) are standard approaches to prevent sample redundancy in batch queries.
- **Hybrid / Filtered Active Learning**: Candidate filtering approaches (selecting top-$M$ uncertain candidates and applying farthest-first diversity) have been explored in general active learning (e.g., Badge by Ash et al., 2019; Cluster-Margin by Citovsky et al., 2021).
- **Temperature Calibration**: Temperature scaling (Guo et al., 2017) is a standard post-hoc calibration technique for neural networks. Its integration into active learning loops has been examined in general computer vision (e.g., Mind the Gap, Krishnan et al., 2020), but rarely under frozen 3D-ResNet foundation representations for hyperspectral imagery.
- **Cross-Sensor Spectral Alignment**: Resampling spectral bands via 1D linear interpolation (Rasti et al., 2018) and CORAL feature alignment (Sun et al., 2016) are established domain adaptation primitives.

---

## 2. Distinction & Novelty Assessment

> [!NOTE]
> **No Claim of Absolute Novelty**: No claim is made that individual components (least confidence, farthest-first traversal, temperature scaling, 1D spectral resampling, or class weighting) are individually novel. Each component is derived from established literature.

### What is Genuinely Distinct in Our Contribution?

1. **Frozen Foundation Model Paradigm for HSI Active Learning**:
   Unlike traditional HSI active learning frameworks that either train small CNNs/SVMs from scratch (Tuia et al., 2011) or end-to-end fine-tune deep backbones on tiny target sets (risking catastrophic overfitting and huge computational cost), our framework leverages a **100% frozen SpectralEarth pretrained `spec_resnet50` backbone** (24.2M parameters) and updates only a lightweight MLP classifier head (533K parameters).

2. **Systematic Combination & Interaction Analysis (SCUD-AL Hypothesis)**:
   We evaluate the unified integration of **Sensor-Aware Calibrated Uncertainty-Diversity Active Learning (SCUD-AL)**. We empirically prove that blindly stacking all theoretically positive modifications (dropout + class weights + labeled pool diversity) **hurts** performance due to negative gradient interactions, and we provide controlled selection criteria.

3. **Zero-Overhead Feature Caching**:
   By exploiting the 100% frozen property of the SpectralEarth backbone, feature embeddings for the dataset pool are cached, reducing per-round feature extraction time from **7.73 seconds to 0.09 seconds (84.13x - 2845.2x speedup)**, making multi-seed active learning benchmark runs computationally feasible.

4. **Multi-Sensor Cross-Domain Transfer under Frozen Backbone Constraints**:
   Demonstrating cross-sensor (AVIRIS 200/204 bands $\to$ ROSIS 103 bands) and cross-dataset transfer using 1D deterministic spectral resampling combined with hidden representation transfer (FC1 weights) without fine-tuning backbone weights.

---

## 3. Summary Statement
Our contribution is an **experimentally validated, computationally efficient, publication-ready engineering and research system** that rigorously benchmarks calibrated uncertainty, diversity, labeled-pool awareness, and cross-sensor adaptation under strict frozen foundation model constraints across 3 standard HSI datasets and 5 random seeds.
