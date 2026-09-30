# LaTeX Figures and Screenshots Manifest

This document provides the exact list of images/screenshots required by `paper.tex`, the exact filename each screenshot must be saved as, what visual content it represents, and the corresponding LaTeX code to include it.

---

## Required Figures Summary Table

| Figure # in Paper | Exact Filename to Save As | Description / Visual Content | LaTeX Code Location | Source in Workspace |
| :---: | :--- | :--- | :--- | :--- |
| **Figure 1** | `downstream_tasks_overview.png` | **Full System Architecture Diagram**: Shows input hyperspectral patch, frozen SpecResNet50 backbone, 2048-D feature cache, 2-layer MLP classifier, temperature scaling, active query engine, and cross-sensor adaptation. | Lines 120–124 | `d:\Projects\spectral_earth\downstream_tasks_overview.png` |
| **Figure 2(a)** | `fig_oa_curve.png` | **Overall Accuracy vs. Labeled Budget**: Multi-round learning curves comparing Random, Least Confidence, and LC + Farthest Diversity on Indian Pines, Salinas, and Pavia University. | Lines 430–435 | `d:\Projects\spectral_earth\logs\al_experiments\figures\oa_vs_labeled.png` |
| **Figure 2(b)** | `fig_multiseed_curves.png` | **5-Seed Mean $\pm$ Std Confidence Bands**: Shaded variance curves across seeds 42, 43, 44, 45, and 46 demonstrating stability of the calibrated query engine. | Lines 436–440 | `d:\Projects\spectral_earth\logs\al_experiments\figures\multiseed_mean_std_curves.png` |
| **Figure 2(c)** | `fig_ece_calibration.png` | **Logit Calibration (ECE Reduction)**: Bar/curve chart showing Expected Calibration Error (ECE) dropping by 43.8% to 50.3% after post-hoc temperature scaling. | Lines 441–446 | `d:\Projects\spectral_earth\logs\al_experiments\figures\ece_before_after_calibration.png` |
| **Figure 3** | `fig_cross_sensor_transfer.png` | **Cross-Sensor Adaptation Curves**: Target sensor (ROSIS Pavia University, 103 bands) learning progression initialized from AVIRIS source representations. | Lines 647–652 | `d:\Projects\spectral_earth\logs\visualizations\cross_sensor\few_shot_transfer_oa_curves.png` |

---

## Detailed Visual Descriptions & LaTeX Integration

### 1. Figure 1: System Architecture Diagram
- **Filename:** `downstream_tasks_overview.png`
- **Aspect Ratio:** Wide / Banner ($\sim 16:9$ or $2:1$)
- **LaTeX Code in `paper.tex`:**
```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=0.94\textwidth]{downstream_tasks_overview.png}
\caption{System architecture of the proposed Adaptive Label-Efficient Hyperspectral Framework, detailing frozen foundation backbone feature extraction, in-memory feature caching, lightweight classification, validation temperature calibration, active query selection, and cross-sensor deterministic adaptation.}
\label{fig:framework_architecture}
\end{figure*}
```
- **What it shows:**
  1. Input hyperspectral scene patch ($B \times 15 \times 15$).
  2. Frozen SpectralEarth 3D-ResNet backbone (24.2M parameters).
  3. Precomputed 2048-dimensional feature cache ($84.13\times$ speedup).
  4. Lightweight 2-layer MLP head (533K parameters, 2.15% trainable).
  5. Post-hoc validation temperature scaling module.
  6. Model B Query Engine (calibrated least-confidence candidate filtering + farthest-first diversity traversal).
  7. Human annotation loop and rapid classifier retraining (~0.150 s).
  8. Deterministic 1D linear spectral resampler (`SpectralResampler1D`) for cross-sensor adaptation.

---

### 2. Figure 2: Tripartite Multi-Seed Performance Progression
This composite figure contains three subfigures side by side:

```latex
\begin{figure*}[!t]
\centering
\begin{subfigure}[b]{0.32\textwidth}
    \includegraphics[width=\textwidth]{fig_oa_curve.png}
    \caption{Overall Accuracy vs. Labeled Budget}
    \label{fig:oa_curves}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
    \includegraphics[width=\textwidth]{fig_multiseed_curves.png}
    \caption{5-Seed Mean $\pm$ Std Performance}
    \label{fig:multiseed}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.32\textwidth}
    \includegraphics[width=\textwidth]{fig_ece_calibration.png}
    \caption{ECE Reduction via Temperature Scaling}
    \label{fig:ece_calibration}
\end{subfigure}
\caption{Empirical active learning performance curves across Indian Pines, Salinas, and Pavia University under 5 random seeds (42, 43, 44, 45, 46).}
\label{fig:al_performance}
\end{figure*}
```

#### Subfigure 2(a): Overall Accuracy vs. Labeled Budget
- **Filename:** `fig_oa_curve.png`
- **What it shows:** The x-axis represents the number of labeled samples (from initial 160 up to 260 for Indian Pines/Salinas, 90 to 190 for Pavia University). The y-axis shows Overall Accuracy (OA %). Curves illustrate `Calibrated LC + Farthest Diversity` outpacing `Random Baseline` and `Least Confidence`.

#### Subfigure 2(b): 5-Seed Mean $\pm$ Std Performance
- **Filename:** `fig_multiseed_curves.png`
- **What it shows:** Mean accuracy curve with shaded $\pm 1$ standard deviation ribbons across seeds 42, 43, 44, 45, and 46. Demonstrates that candidate filtering with farthest-first diversity significantly reduces cross-seed variance compared to pure uncertainty sampling.

#### Subfigure 2(c): ECE Reduction via Temperature Scaling
- **Filename:** `fig_ece_calibration.png`
- **What it shows:** Comparison of Expected Calibration Error before vs. after post-hoc temperature scaling ($T \approx 1.3 - 1.5$), illustrating a dramatic 43.8% to 50.3% drop in calibration error across all three datasets.

---

### 3. Figure 3: Cross-Sensor Adaptation Curves
- **Filename:** `fig_cross_sensor_transfer.png`
- **Aspect Ratio:** Standard ($4:3$ or column width)
- **LaTeX Code in `paper.tex`:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.88\columnwidth]{fig_cross_sensor_transfer.png}
\caption{Few-shot active learning adaptation accuracy curves on target sensor (ROSIS Pavia University, 103 bands) initialized from AVIRIS source representations.}
\label{fig:cross_sensor_curves}
\end{figure}
```
- **What it shows:**
  1. Zero-shot transfer starting point (49.84% OA for IP $\to$ Pavia; 54.12% for Salinas $\to$ Pavia) enabled by `SpectralResampler1D`.
  2. Steep target active learning adaptation curve climbing rapidly to 61.25%--61.80% OA within 190 target labels.
  3. Confirms that source FC1 hidden layer representations accelerate target convergence.

---

## Directory Setup for LaTeX Compilation

When compiling `paper.tex` (either locally or on Overleaf), place the 5 image files in the same directory as `paper.tex` and `references.bib`:

```
project_folder/
├── paper.tex
├── references.bib
├── downstream_tasks_overview.png   <-- Figure 1
├── fig_oa_curve.png                <-- Figure 2(a)
├── fig_multiseed_curves.png        <-- Figure 2(b)
├── fig_ece_calibration.png         <-- Figure 2(c)
└── fig_cross_sensor_transfer.png   <-- Figure 3
```

All 5 files are already copied with these exact filenames in:
`d:\Projects\spectral_earth\spectral_earth_latex_package\`
and are included in the downloadable archive:
`d:\Projects\spectral_earth\spectral_earth_latex_report.zip`
