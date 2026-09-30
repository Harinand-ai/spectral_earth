# Cross-Sensor & Cross-Dataset Transfer Feasibility Audit Report

This audit assesses the scientific validity, technical feasibility, and architectural requirements for extending the **SpectralEarth Active Learning Framework** to cross-sensor and cross-dataset transfer experiments.

---

## 1. Feasibility Decision

### Decision: **B. FEASIBLE WITH A SMALL ISOLATED ADAPTER**

> [!NOTE]
> Cross-dataset transfer (Indian Pines $\leftrightarrow$ Salinas) and cross-sensor transfer (AVIRIS $\leftrightarrow$ ROSIS) can be executed using an isolated 1D spectral resampling adapter without unfreezing SpectralEarth, modifying Model A/Model B, or altering the frozen active learning pipeline.

---

## 2. Sensor & Wavelength Metadata Audit

| Dataset | Instrument / Sensor | Platform | Band Count | Spectral Range ($\mu\text{m}$) | Spatial Resolution |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Indian Pines** | AVIRIS | Airborne | 200 | $0.40 - 2.50\ \mu\text{m}$ (400–2500 nm) | 20 m / pixel |
| **Salinas** | AVIRIS | Airborne | 204 | $0.40 - 2.50\ \mu\text{m}$ (400–2500 nm) | 3.7 m / pixel |
| **Pavia University** | ROSIS | Airborne | 103 | $0.43 - 0.86\ \mu\text{m}$ (430–860 nm) | 1.3 m / pixel |

### Scientific Terminology Assessment
- **Indian Pines $\leftrightarrow$ Salinas**: **Cross-Dataset / Cross-Domain Transfer** (Same sensor family AVIRIS, identical spectral range $0.4 - 2.5\ \mu\text{m}$, minor band differences 200 vs 204).
- **Indian Pines / Salinas $\leftrightarrow$ Pavia University**: **Cross-Sensor & Cross-Domain Transfer** (AVIRIS vs ROSIS, distinct spectral coverage $0.4 - 2.5\ \mu\text{m}$ vs $0.43 - 0.86\ \mu\text{m}$, distinct band counts 200 vs 103).

---

## 3. Technical & Architectural Audit

1. **SpectralEarth Backbone Capabilities**:
   - `SpectralAdapter` (`src/backbones/spectral_adapter.py`) uses 3D convolutions followed by `AdaptiveAvgPool3d((1, None, None))`.
   - The architecture can dynamically accept input tensors with varying channel counts ($C=200$, $C=204$, $C=103$) without tensor shape mismatch errors.

2. **Spectral Alignment Requirement**:
   - To ensure physical wavelength consistency when passing data into frozen backbone kernels trained on 200 bands, target spectral bands should be resampled via 1D linear interpolation to match source wavelength channels.

3. **Isolated Preprocessing Adapter Design**:
   $$\text{Target Patch } [B, C_{\text{target}}, H, W] \xrightarrow{\text{1D Wavelength Resampling}} [B, C_{\text{source}}, H, W] \xrightarrow{\text{Frozen SpectralEarth}} \text{Embeddings } [B, 2048]$$

4. **Proposed Transfer Protocol**:
   - **Step 1**: Train classifier head on Source Dataset labeled pool ($L_{\text{source}}$).
   - **Step 2**: Apply isolated 1D spectral resampling to Target Dataset.
   - **Step 3**: Perform Zero-Shot evaluation or Few-Shot Active Learning adaptation ($LC+FD$) on Target Dataset.

---

## 4. Risks & Scientific Validity

- **Wavelength Overlap**: ROSIS (Pavia University) covers $430 - 860$ nm, which corresponds to the visible/near-infrared subset of AVIRIS (bands 1–45). Resampling ROSIS to full AVIRIS range requires zero-padding SWIR bands.
- **Land Cover Class Alignment**: Target datasets have distinct land-cover classes (e.g. agricultural crops vs urban materials). Head transfer requires class mapping or re-initializing the linear output layer.

---

## 5. Recommendation

This cross-sensor feasibility audit is complete. **No code changes have been made**. Cross-sensor transfer can be implemented as an isolated future extension following review.
