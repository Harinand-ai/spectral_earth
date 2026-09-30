# VISUAL DEMONSTRATION SYSTEM — SPECTRALEARTH ACTIVE LEARNING

**Date**: 2026-08-23  
**Module**: `src/visualization_app/`  
**API Server**: FastAPI (`app.py`)  
**Frontend UI**: Interactive Single-Page Application (`static/index.html`, `static/app.js`)  

---

## 1. Executive Summary

The SpectralEarth Visualization Application is an interactive, web-based demonstration system designed to communicate the active learning process to researchers, reviewers, and non-expert stakeholders. It translates backend active learning operations into an intuitive, 11-step interactive narrative.

---

## 2. The 11-Step Interactive Active Learning Story

The web dashboard guides users through the complete active learning cycle step by step:

```
[STEP 1: Scene Selection]
       ↓
[STEP 2: Initial Seed Pool] (10 samples/class)
       ↓
[STEP 3: Model Prediction Map]
       ↓
[STEP 4: Uncertainty Heatmap] (Least Confidence / Entropy)
       ↓
[STEP 5: Candidate Selection] (LC + Farthest Diversity, k=20)
       ↓
[STEP 6: Sample-Level Inspection & Explanation] ("Why was this sample selected?")
       ↓
[STEP 7: Oracle Human Annotation] (Simulated Ground Truth Labeling)
       ↓
[STEP 8: Classifier Retraining] (2-layer MLP Head, 25 Epochs)
       ↓
[STEP 9: Prediction Map BEFORE vs AFTER]
       ↓
[STEP 10: Multi-Round Progression] (Round 0 → Round 5)
       ↓
[STEP 11: Accuracy & Label-Efficiency Curves] (OA, AA, Macro F1, ECE)
```

---

## 3. "Show Me How an Unlabeled Sample Becomes Labeled" Feature

A core visual requirement of the system is displaying the explicit lifecycle of individual candidate pixels.

### Pixel Lifecycle Pipeline:
1. **Unlabeled Inspection**: User selects an unlabeled pixel location $(r, c)$. Dashboard displays:
   - $15 \times 15$ RGB Spatial Patch centered at $(r, c)$.
   - 1D Spectral Reflectance Signature across available bands.
   - Initial Prediction Probability Distribution ($\hat{y} = \text{Soybean-mintill}$, $\text{Confidence} = 0.163$).
   - Uncertainty Score ($\text{Least Confidence} = 0.837$).
2. **Query Engine Selection**: QueryEngine selects candidate rank $\#1$:
   - *Explanatory Reason*: High predictive uncertainty ($\text{LC} = 0.837$, top 0.5%) + High embedding space distance ($\text{Diversity} = 0.202$, top 36.6%).
3. **Oracle Annotation**: Ground truth label $y = \text{Soybean-mintill}$ is revealed and transferred to the Labeled Pool ($L$).
4. **Classifier Retraining**: 2-layer MLP head retrains for 25 epochs on updated pool $L$.
5. **Retrained Confidence Jump**: Re-evaluating pixel $(r, c)$ displays updated confidence:
   - $\hat{y} = \text{Soybean-mintill}$, $\text{Confidence} = 0.910$ ($\mathbf{+74.7\%}$ confidence jump).

---

## 4. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves main SPA HTML dashboard (`index.html`). |
| `/api/datasets` | `GET` | Lists available datasets (Indian Pines, Salinas, Pavia University). |
| `/api/session/select_dataset` | `POST` | Initializes active learning session for selected dataset & seed. |
| `/api/scene_data` | `GET` | Returns scene metadata, RGB image composite, GT labels, spatial mask. |
| `/api/predictions_and_uncertainty` | `GET` | Computes prediction map, confidence map, and spatial uncertainty heatmap. |
| `/api/query_selection` | `POST` | Runs LC + Farthest Diversity query selection with sample explanations. |
| `/api/sample_inspection/{index}` | `GET` | Returns spatial patch, spectral signature, class probabilities, and query reason. |
| `/api/annotate_and_retrain` | `POST` | Simulates oracle annotation, updates pool, retrains MLP head, returns metrics. |
| `/api/before_after_maps` | `GET` | Returns prediction map prior to retraining vs current prediction map. |
| `/api/round_progression` | `GET` | Returns multi-round metrics history across active learning rounds. |
| `/api/cross_sensor_data` | `GET` | Serves AVIRIS $\to$ ROSIS 1D spectral resampling alignment and transfer metrics. |
| `/api/label_efficiency_data` | `GET` | Returns multi-strategy learning curves and label savings threshold table. |
| `/api/export_experiment` | `POST` | Exports current session results to JSON and CSV under `logs/al_experiments/`. |
| `/api/generate_report` | `GET` | Renders self-contained HTML/PDF experiment report. |

---

## 5. Verification & Launching Instructions

```bash
# Launch interactive web application server
python run_visualization_app.py

# Access dashboard in web browser:
# http://localhost:8000
```
