// GLOBAL APPLICATION STATE
let currentDataset = "Indian Pines";
let currentStrategy = "margin";
let currentMode = "oracle"; // "oracle" or "human"
let targetOA = 70.0;
let sceneData = null;
let currentMapType = "rgb"; // "rgb", "pool", "pred", "unc"
let currentUncertaintyMethod = "margin";
let lastPredictionsData = null;
let querySelectionData = null;
let isDatasetLoading = false;
let hasPredicted = false;
let hasQueried = false;
let currentSceneViewMode = '2d';

// 3D Post-Annotation Visualization Layer Toggles
let show3DPredictions = true;
let hasRetrained = false;
let show3DHuman = true;
let show3DPseudo = true;
let show3DQueried = true;

function hexToRgbArray(hex) {
    if (!hex) return [37, 99, 235];
    if (Array.isArray(hex)) return hex;
    if (typeof hex === 'string' && hex.startsWith('rgb')) {
        const parts = hex.match(/\d+/g);
        if (parts && parts.length >= 3) return parts.slice(0, 3).map(Number);
    }
    let c = String(hex).replace('#', '').trim();
    if (c.length === 3) c = c[0] + c[0] + c[1] + c[1] + c[2] + c[2];
    const num = parseInt(c, 16);
    if (isNaN(num)) return [37, 99, 235];
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function hexToRgbString(hex) {
    if (!hex) return 'rgb(37, 99, 235)';
    if (typeof hex === 'string' && hex.startsWith('rgb')) return hex;
    const [r, g, b] = hexToRgbArray(hex);
    return `rgb(${r}, ${g}, ${b})`;
}
const hexToRgb = hexToRgbString;

// TOAST NOTIFICATIONS HELPER
function showToast(message, type = "success", duration = 3200) {
    const container = document.getElementById("toastContainer");
    if (!container) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "✓" : type === "info" ? "ℹ" : type === "warning" ? "⚠" : "✕";
    toast.innerHTML = `<span style="font-weight:700; color:${type==='success'?'var(--success)':type==='warning'?'var(--warning)':'var(--primary)'};">${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(12px) scale(0.95)";
        toast.style.transition = "all 0.25s ease";
        setTimeout(() => toast.remove(), 250);
    }, duration);
}

// BUTTON LOADING HELPER
function setButtonLoading(btn, isLoading, loadingText = "Loading...") {
    if (!btn) return;
    if (isLoading) {
        btn.dataset.originalText = btn.innerHTML;
        btn.classList.add("btn-loading");
        btn.disabled = true;
        btn.innerHTML = `<span class="btn-spinner"></span> ${loadingText}`;
    } else {
        btn.classList.remove("btn-loading");
        btn.disabled = false;
        if (btn.dataset.originalText) {
            btn.innerHTML = btn.dataset.originalText;
        }
    }
}

// PROGRESSIVE DISCLOSURE RESET
function resetProgressiveDisclosure() {
    hasPredicted = false;
    hasRetrained = false;
    hasQueried = false;
    querySelectionData = null;

    const sceneEmpty = document.getElementById("sceneEmptyState");
    const sceneActive = document.getElementById("sceneActiveContent");
    const sceneControls = document.getElementById("sceneMapControls");
    if (sceneEmpty) sceneEmpty.style.display = "none";
    if (sceneActive) sceneActive.style.display = "block";
    if (sceneControls) sceneControls.style.display = "none";

    const infoEl = document.getElementById("scene3dSelectedInfo");
    if (infoEl) {
        infoEl.innerHTML = '<span style="color:var(--primary); font-weight:600; margin-right:6px;">✦</span> Click or hover any of the 20 queried sample markers (<strong>#1–#20</strong>) in 3D space to inspect rationale and open full sample diagnostics.';
    }
    const badgeEl = document.getElementById("scene3dSamplesBadge");
    if (badgeEl) {
        badgeEl.innerHTML = `Query Informative Samples to display 20 AL candidate locations in 3D`;
    }
    stopScene3DAutoRotate();

    const progEmpty = document.getElementById("progressionEmptyState");
    const progActive = document.getElementById("progressionActiveContent");
    if (progEmpty) progEmpty.style.display = "none";
    if (progActive) progActive.style.display = "block";

    const tableEmpty = document.getElementById("tableEmptyState");
    const tableWrapper = document.getElementById("queryTableWrapper");
    if (tableEmpty) tableEmpty.style.display = "flex";
    if (tableWrapper) tableWrapper.style.display = "none";

    const btnQuery = document.getElementById("btnRunQuery");
    if (btnQuery) {
        btnQuery.disabled = false;
        btnQuery.className = "btn btn-primary";
        btnQuery.innerHTML = "1. Query Samples (k=20)";
    }

    const btnRetrain = document.getElementById("btnSubmitAnnotate");
    if (btnRetrain) {
        btnRetrain.disabled = true;
        btnRetrain.className = "btn btn-secondary disabled-stage";
        btnRetrain.innerHTML = currentMode === "human" ? "2. Submit Annotations & Retrain" : "2. Retrain Classifier";
    }
}

// CHARTS & CANVAS
let progressionChart = null;
let resamplerChart = null;
let benchmarkChart = null;
let labelEfficiencyChart = null;
let sampleSpectrumChart = null;
let sampleProbChart = null;
let currentQueryBatchSize = 20;

// 3D/2D Interactive Chart Visualization State
const chartModes = {
    progression: '2d',
    resampler: '2d',
    benchmark: '2d',
    labelEff: '2d',
    sampleSpectrum: '2d',
    csCurves: '2d'
};
let lastProgressionHistory = null;
let lastResamplerData = null;
let lastCsCurvesData = null;
let lastBenchmarkParams = { dsName: 'Indian Pines', metricView: 'oa' };
let lastLabelEfficiencyCurves = null;
let lastSampleData = null;

function setChartMode(chartName, mode) {
    chartModes[chartName] = mode;
    const cap = capitalizeFirst(chartName);
    const btn3d = document.getElementById(`btn${cap}3D`) || document.getElementById(`btn${cap.replace(/s$/, '')}3D`);
    const btn2d = document.getElementById(`btn${cap}2D`) || document.getElementById(`btn${cap.replace(/s$/, '')}2D`);
    if (btn3d && btn2d) {
        if (mode === '3d') {
            btn3d.classList.add('active');
            btn2d.classList.remove('active');
        } else {
            btn2d.classList.add('active');
            btn3d.classList.remove('active');
        }
    }

    if (chartName === 'progression' && lastProgressionHistory) {
        renderProgressionChart(lastProgressionHistory);
    } else if (chartName === 'csCurves' && lastCsCurvesData) {
        renderCrossSensorCurves(lastCsCurvesData, window._lastCsMeta);
    } else if (chartName === 'resampler' && lastResamplerData) {
        render1DSpectralResamplerChart(lastResamplerData);
    } else if (chartName === 'benchmark' && lastBenchmarkParams) {
        renderBenchmarkChart(lastBenchmarkParams.dsName, lastBenchmarkParams.metricView);
    } else if (chartName === 'labelEff' && lastLabelEfficiencyCurves) {
        renderLabelEfficiencyChart(lastLabelEfficiencyCurves);
    } else if (chartName === 'sampleSpectrum' && lastSampleData) {
        renderSampleSpectrumChart(lastSampleData);
    }
}

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function resetChartCamera(chartId) {
    const el = document.getElementById(chartId);
    if (el && window.Plotly) {
        Plotly.relayout(el, {
            'scene.camera': {
                eye: { x: 1.6, y: -1.7, z: 1.2 },
                center: { x: 0, y: 0, z: -0.1 },
                up: { x: 0, y: 0, z: 1 }
            }
        });
    }
}

window.setChartMode = setChartMode;
window.resetChartCamera = resetChartCamera;

// VERIFIED 5-SEED BENCHMARK DATA (SEED 42-46)
const BENCHMARK_DATA = {
    "Indian Pines": [
        { method: "Random Sampling", oa: "60.34 ± 2.43", aa: "67.71 ± 1.20", f1: "0.5739 ± 0.0165", kappa: "0.5474 ± 0.0250", isBest: false, stratKey: "random" },
        { method: "Least Confidence", oa: "58.53 ± 3.38", aa: "66.69 ± 1.75", f1: "0.5678 ± 0.0261", kappa: "0.5322 ± 0.0350", isBest: false, stratKey: "lc" },
        { method: "LC + Farthest Diversity", oa: "60.73 ± 2.88", aa: "68.14 ± 2.80", f1: "0.5993 ± 0.0197", kappa: "0.5552 ± 0.0316", isBest: true, stratKey: "lc_fd" }
    ],
    "Salinas": [
        { method: "Random Sampling", oa: "71.25 ± 1.50", aa: "74.68 ± 2.00", f1: "0.7222 ± 0.0118", kappa: "0.6808 ± 0.0164", isBest: false, stratKey: "random" },
        { method: "Least Confidence", oa: "70.48 ± 2.29", aa: "77.00 ± 2.57", f1: "0.7332 ± 0.0218", kappa: "0.6752 ± 0.0244", isBest: false, stratKey: "lc" },
        { method: "LC + Farthest Diversity", oa: "72.19 ± 1.61", aa: "77.13 ± 2.65", f1: "0.7459 ± 0.0232", kappa: "0.6925 ± 0.0186", isBest: true, stratKey: "lc_fd" }
    ],
    "Pavia University": [
        { method: "Random Sampling", oa: "65.51 ± 1.96", aa: "64.14 ± 2.34", f1: "0.5987 ± 0.0205", kappa: "0.5521 ± 0.0253", isBest: false, stratKey: "random" },
        { method: "Least Confidence", oa: "55.71 ± 1.70", aa: "62.90 ± 2.18", f1: "0.5601 ± 0.0082", kappa: "0.4513 ± 0.0187", isBest: false, stratKey: "lc" },
        { method: "LC + Farthest Diversity", oa: "60.01 ± 1.54", aa: "65.97 ± 1.88", f1: "0.6041 ± 0.0132", kappa: "0.5021 ± 0.0172", isBest: true, stratKey: "lc_fd" }
    ]
};

/* =========================================================================
   INTRODUCTORY ANIMATION OVERLAY CONTROLLER
   ========================================================================= */
let introTimer = null;
let introStepTimer = null;


// =========================================================================
// GRAPH GENERATING RADAR ANIMATION CONTROLLER
// =========================================================================
function showGraphGeneratingAnimation(containerEl, message = "Generating visualization...", submessage = "SpectralEarth Decoupled Pipeline") {
    if (!containerEl) return;
    if (getComputedStyle(containerEl).position === 'static') {
        containerEl.style.position = 'relative';
    }
    let overlay = containerEl.querySelector('.graph-generating-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'graph-generating-overlay';
        overlay.innerHTML = `
            <div class="graph-spinner-wrapper">
                <div class="graph-spinner-ring"></div>
                <div class="graph-spinner-inner"></div>
                <div class="graph-spinner-core"></div>
            </div>
            <div class="graph-generating-text">${message}</div>
            <div class="graph-generating-subtext">${submessage}</div>
        `;
        containerEl.appendChild(overlay);
    } else {
        const textEl = overlay.querySelector('.graph-generating-text');
        if (textEl) textEl.textContent = message;
        const subEl = overlay.querySelector('.graph-generating-subtext');
        if (subEl) subEl.textContent = submessage;
        overlay.classList.remove('fade-out');
    }
}

function hideGraphGeneratingAnimation(containerEl) {
    if (!containerEl) return;
    const overlay = containerEl.querySelector('.graph-generating-overlay');
    if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => {
            if (overlay && overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }, 280);
    }
}

function initIntroAnimation(forceReplay = false) {
    // Landing animation permanently disabled in favor of professional instant dashboard
    return;
}
function closeIntroAnimation() {
    return;
}


document.addEventListener("DOMContentLoaded", () => {
    initIntroAnimation();
    initTabNavigation();
    initEventListeners();
    initGuidedEngine();
    loadDataset(currentDataset, false);
    switchDatasetExplorerView("Indian Pines");
    loadBenchmarkData();
    loadCrossSensorPair("Indian Pines -> Salinas");
});

/* =========================================================================
   NAVIGATION & TABS (6 REORGANIZED SECTIONS)
   ========================================================================= */
function initTabNavigation() {
    const tabs = document.querySelectorAll(".nav-bar .nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add("active");

            if (window.tutorialEngine && window.tutorialEngine.isActive) {
                window.tutorialEngine.notifyAction("TAB_CHANGED", { tabId: targetId });
            }

            // Auto-resize Plotly graphs when tab becomes visible
            setTimeout(() => {
                ['progressionChart', 'resamplerChart', 'benchmarkChart', 'labelEfficiencyChart', 'csCurvesChart', 'confusionMatrixChart'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el && window.Plotly && el.offsetHeight > 0) {
                        Plotly.Plots.resize(el);
                    }
                });
            }, 60);

            // Dynamic Tab Data Loading
            if (targetId === "home-demo-tab") {
                setTimeout(() => {
                    renderSpatialCanvas();
                    const el = document.getElementById("progressionChart");
                    if (el && window.Plotly && el.offsetHeight > 0) Plotly.Plots.resize(el);
                }, 50);
            } else if (targetId === "cross-sensor-tab") {
                const pairSel = document.getElementById("transferPairSelect");
                const pKey = pairSel ? pairSel.value : "Indian Pines -> Salinas";
                loadCrossSensorPair(pKey);
            } else if (targetId === "benchmarks-tab") {
                loadBenchmarkData();
            } else if (targetId === "datasets-tab") {
                switchDatasetExplorerView(window._currentExplorerDs || "Indian Pines");
            }
        });
    });
}

/* =========================================================================
   EVENT LISTENERS (SINGLE AUTHORITATIVE HANDLERS)
   ========================================================================= */
function initEventListeners() {
    // Replay Introductory Animation
    const btnReplayIntro = document.getElementById("btnReplayIntro");
    if (btnReplayIntro) {
        btnReplayIntro.addEventListener("click", () => {
            initIntroAnimation(true);
        });
    }

    // Benchmark Historical Baselines Toggle
    const chkShowAll = document.getElementById("chkShowAllMethods");
    if (chkShowAll) {
        chkShowAll.addEventListener("change", () => {
            renderBenchmarkTab();
        });
    }

    // Single Authoritative Dataset Switcher
    const dsSel = document.getElementById("datasetSelect");
    if (dsSel) {
        dsSel.addEventListener("change", (e) => {
            currentDataset = e.target.value;
            loadDataset(currentDataset, false);
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
        dsSel.addEventListener("click", () => {
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
    }

    // Strategy switcher
    const stratSel = document.getElementById("strategySelect");
    if (stratSel) {
        stratSel.addEventListener("change", (e) => {
            currentStrategy = e.target.value;
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
        stratSel.addEventListener("click", () => {
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
    }

    // Mode Toggle Switcher
    const modeSel = document.getElementById("modeToggleSelect");
    if (modeSel) {
        modeSel.addEventListener("change", async (e) => {
            currentMode = e.target.value;
            await setSessionMode(currentMode);
            if (window.tutorialEngine && window.tutorialEngine.isActive) {
                if (window.tutorialEngine.currentState === "PREPARE" || window.tutorialEngine.currentState === "ANNOTATE") {
                    window.tutorialEngine.renderState();
                }
            }
        });
        modeSel.addEventListener("click", () => {
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
    }

    // Query Size Selector (Configurable k)
    const kSel = document.getElementById("queryBatchSelect");
    if (kSel) {
        kSel.addEventListener("change", (e) => {
            const kVal = parseInt(e.target.value) || 20;
            currentQueryBatchSize = kVal;
            const btnQuery = document.getElementById("btnRunQuery");
            if (btnQuery) {
                btnQuery.innerHTML = `1. Query Samples (k=${kVal})`;
            }
            if (window.tutorialEngine && window.tutorialEngine.isActive) {
                window.tutorialEngine.sessionMetrics.querySize = kVal;
                if (window.tutorialEngine.currentState === "PREPARE" || window.tutorialEngine.currentState === "QUERY") {
                    window.tutorialEngine.renderState();
                }
            }
        });
        kSel.addEventListener("click", () => {
            if (window.tutorialEngine && window.tutorialEngine.isActive && window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        });
    }

    // Pseudo-labeling toggle
    const pseudoSel = document.getElementById("pseudoToggleSelect");
    if (pseudoSel) {
        pseudoSel.addEventListener("change", async (e) => {
            const isEnabled = (e.target.value === "on");
            try {
                const res = await fetch("/api/session/set_pseudo_labeling", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        use_pseudo_labeling: isEnabled,
                        pseudo_threshold: 0.95,
                        pseudo_weight: 0.2,
                        max_pseudo_per_class: 15
                    }),
                });
                await res.json();
                const pDot = document.getElementById("pseudoStatusDot");
                const pLabel = document.getElementById("pseudoStatusLabel");
                if (pDot && pLabel) {
                    pDot.style.display = isEnabled ? "inline" : "none";
                    pLabel.style.display = isEnabled ? "inline" : "none";
                }
                showToast(isEnabled ? "Confidence Pseudo-Labeling Enabled (tau=0.95, lambda=0.2, class_cap=15)" : "Pseudo-Labeling Disabled", isEnabled ? "success" : "info");
            } catch (err) {
                console.error("Error setting pseudo labeling:", err);
            }
        });
    }

    // Target OA setter
    const btnSetTarget = document.getElementById("btnSetTarget");
    if (btnSetTarget) {
        btnSetTarget.addEventListener("click", async () => {
            const val = parseFloat(document.getElementById("targetOaInput").value) || 70.0;
            targetOA = val;
            await setSessionTargetOA(targetOA);
            if (window.tutorialEngine && window.tutorialEngine.isActive) {
                window.tutorialEngine.notifyAction("TARGET_SET", {
                    target_oa: targetOA,
                    achieved_oa: document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "",
                });
            }
        });
    }

    // Map Toggle buttons
    if (document.getElementById("btnMapRGB")) document.getElementById("btnMapRGB").addEventListener("click", () => switchMap("rgb"));
    if (document.getElementById("btnMapPool")) document.getElementById("btnMapPool").addEventListener("click", () => switchMap("pool"));
    if (document.getElementById("btnMapPred")) document.getElementById("btnMapPred").addEventListener("click", () => switchMap("pred"));
    if (document.getElementById("btnMapUnc")) document.getElementById("btnMapUnc").addEventListener("click", () => switchMap("unc"));

    // Initial Prediction Button
    const btnPredict = document.getElementById("btnInitialPredict");
    if (btnPredict) {
        btnPredict.addEventListener("click", async () => {
            setButtonLoading(btnPredict, true, "Computing Baseline...");
            try {
                await fetchAndRenderPredictionsAndUncertainty();
                await fetchRoundProgression();
                switchMap("pred");

                hasPredicted = true;
                const sceneEmpty = document.getElementById("sceneEmptyState");
                const sceneActive = document.getElementById("sceneActiveContent");
                const sceneControls = document.getElementById("sceneMapControls");
                if (sceneEmpty) sceneEmpty.style.display = "none";
                if (sceneActive) sceneActive.style.display = "block";
                if (sceneControls && (currentSceneViewMode === '2d' || !currentSceneViewMode)) sceneControls.style.display = "flex";
                setSceneViewMode(currentSceneViewMode || '2d');

                const progEmpty = document.getElementById("progressionEmptyState");
                const progActive = document.getElementById("progressionActiveContent");
                if (progEmpty) progEmpty.style.display = "none";
                if (progActive) progActive.style.display = "block";

                const btnQuery = document.getElementById("btnRunQuery");
                if (btnQuery) {
                    btnQuery.disabled = false;
                    btnQuery.classList.remove("disabled-stage");
                    btnQuery.classList.remove("btn-secondary");
                    btnQuery.classList.add("btn-primary");
                }

                btnPredict.classList.remove("btn-primary");
                btnPredict.classList.add("btn-secondary");

                const oaVal = document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "52.40%";
                const f1Val = document.getElementById("metricF1") ? document.getElementById("metricF1").innerText : "0.4520";
                showToast(`Baseline prediction computed (OA: ${oaVal})`, "success");

                if (window.tutorialEngine) {
                    window.tutorialEngine.notifyAction("INITIAL_PREDICTION_DONE", {
                        oa: oaVal,
                        f1: f1Val
                    });
                }
            } catch (err) {
                console.error("Prediction error:", err);
                showToast("Failed to compute baseline prediction", "warning");
            } finally {
                setButtonLoading(btnPredict, false);
            }
        });
    }

    // Step 1: Run Query Selection Button
    const btnQuery = document.getElementById("btnRunQuery");
    if (btnQuery) {
        btnQuery.addEventListener("click", async () => {
            setButtonLoading(btnQuery, true, "Querying Samples...");
            try {
                await runQuerySelection();
                hasQueried = true;

                const tableEmpty = document.getElementById("tableEmptyState");
                const tableWrapper = document.getElementById("queryTableWrapper");
                if (tableEmpty) tableEmpty.style.display = "none";
                if (tableWrapper) tableWrapper.style.display = "block";

                const btnRetrain = document.getElementById("btnSubmitAnnotate");
                if (btnRetrain) {
                    btnRetrain.disabled = false;
                    btnRetrain.classList.remove("disabled-stage");
                    btnRetrain.classList.remove("btn-secondary");
                    btnRetrain.classList.add("btn-primary");
                }

                btnQuery.classList.remove("btn-primary");
                btnQuery.classList.add("btn-secondary");

                const k = querySelectionData ? (querySelectionData.k || currentQueryBatchSize) : currentQueryBatchSize;
                showToast(`Queried ${k} informative samples using ${currentStrategy}`, "success");

                if (window.tutorialEngine) {
                    window.tutorialEngine.notifyAction("QUERY_SAMPLING_DONE", {
                        sampleCount: k,
                        strategy: currentStrategy
                    });
                }
            } catch (err) {
                console.error("Query error:", err);
                showToast("Failed to sample queries", "warning");
            } finally {
                setButtonLoading(btnQuery, false);
                btnQuery.innerHTML = `1. Query Samples (k=${currentQueryBatchSize})`;
            }
        });
    }

    // Step 2: Retrain Classifier / Submit Annotations Button
    const btnRetrain = document.getElementById("btnSubmitAnnotate");
    if (btnRetrain) {
        btnRetrain.addEventListener("click", async () => {
            setButtonLoading(btnRetrain, true, "Retraining Head...");
            try {
                await annotateAndRetrain();

                const oaEl = document.getElementById("metricOA");
                if (oaEl) {
                    oaEl.style.transition = "transform 0.3s ease, color 0.3s ease";
                    oaEl.style.transform = "scale(1.15)";
                    oaEl.style.color = "var(--success)";
                    setTimeout(() => {
                        oaEl.style.transform = "scale(1)";
                        oaEl.style.color = "var(--primary)";
                    }, 1200);
                }

                const newOa = oaEl ? oaEl.innerText : "58.10%";
                showToast(`Retrained classifier! New Overall Accuracy: ${newOa}`, "success");
            } catch (err) {
                console.error("Retrain error:", err);
                showToast("Error retraining classifier", "warning");
            } finally {
                setButtonLoading(btnRetrain, false);
                btnRetrain.innerHTML = currentMode === "human" ? "2. Submit Annotations & Retrain" : "2. Retrain Classifier";
            }
        });
    }

    // Session Export Button
    if (document.getElementById("btnExportSession")) {
        document.getElementById("btnExportSession").addEventListener("click", exportExperimentSession);
    }

    // Cost Estimator Calculator
    if (document.getElementById("btnCalculateCost")) {
        document.getElementById("btnCalculateCost").addEventListener("click", calculateLabelCost);
    }

    // Cross-Sensor Transfer Pair Selector & Live Trial
    const csPair = document.getElementById("transferPairSelect");
    if (csPair) {
        csPair.addEventListener("change", (e) => {
            loadCrossSensorData(e.target.value, true);
        });
    }
    const btnRunAdaptation = document.getElementById("btnRunAdaptationDemo");
    if (btnRunAdaptation) {
        btnRunAdaptation.addEventListener("click", () => {
            runLiveAdaptationTrial();
        });
    }

    // Experimental Evidence Connected Control Bar
    const expDs = document.getElementById("expDatasetSelect");
    const expType = document.getElementById("expTypeSelect");
    const expMetric = document.getElementById("expMetricSelect");

    if (expDs) expDs.addEventListener("change", () => {
        updateExperimentalEvidenceView();
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("EXPERIMENT_TYPE_CHANGED", { type: expType ? expType.value : "benchmark" });
        }
    });
    if (expType) expType.addEventListener("change", () => {
        updateExperimentalEvidenceView();
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("EXPERIMENT_TYPE_CHANGED", { type: expType.value });
        }
    });
    if (expMetric) expMetric.addEventListener("change", () => {
        updateExperimentalEvidenceView();
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("EXPERIMENT_TYPE_CHANGED", { type: expType ? expType.value : "benchmark" });
        }
    });

    // Modal Close
    if (document.getElementById("modalCloseBtn")) {
        document.getElementById("modalCloseBtn").addEventListener("click", () => {
            cleanupHsiExplorer();
            document.getElementById("sampleModal").classList.remove("active");
            if (window.tutorialEngine && window.tutorialEngine.isActive &&
                (window.tutorialEngine.currentState === "INSPECT_MODAL" || window.tutorialEngine.currentState === "ANNOTATE")) {
                window.tutorialEngine.advanceTo("ANNOTATE_RESULT");
            }
        });
    }

    // Report Actions
    if (document.getElementById("btnRefreshReport")) {
        document.getElementById("btnRefreshReport").addEventListener("click", () => {
            const iframe = document.getElementById("reportIframe");
            if (iframe) iframe.src = "/api/generate_report?t=" + new Date().getTime();
        });
    }
    if (document.getElementById("btnDownloadReport")) {
        document.getElementById("btnDownloadReport").addEventListener("click", () => {
            window.open("/api/generate_report", "_blank");
        });
    }
}

/* =========================================================================
   GUIDED ENGINE IMPLEMENTATION
   ========================================================================= */
function initGuidedEngine() {
    const btnToggle = document.getElementById("btnToggleGuided");
    const btnExplain = document.getElementById("btnExplainThis");

    if (btnToggle) {
        btnToggle.addEventListener("click", () => {
            if (window.tutorialEngine && window.tutorialEngine.isActive) {
                window.tutorialEngine.stop();
                btnToggle.innerText = "Guided Tutorial";
                btnToggle.className = "btn btn-primary btn-sm";
            } else if (window.tutorialEngine) {
                window.tutorialEngine.start();
                btnToggle.innerText = "Close Guide";
                btnToggle.className = "btn btn-secondary btn-sm";
            }
        });
    }

    if (btnExplain) {
        btnExplain.addEventListener("click", openExplainModal);
    }

    const closeExpBtn = document.getElementById("explainModalCloseBtn");
    if (closeExpBtn) closeExpBtn.onclick = () => document.getElementById("explainModal").classList.remove("active");
    const okExpBtn = document.getElementById("explainModalOkBtn");
    if (okExpBtn) okExpBtn.onclick = () => document.getElementById("explainModal").classList.remove("active");
    const expModalOverlay = document.getElementById("explainModal");
    if (expModalOverlay) expModalOverlay.onclick = (e) => { if (e.target === expModalOverlay) expModalOverlay.classList.remove("active"); };
}

function openExplainModal() {
    const modal = document.getElementById("explainModal");
    const titleEl = document.getElementById("explainModalTitle");
    const bodyEl = document.getElementById("explainModalBody");
    if (!modal || !bodyEl) return;

    const activeTab = document.querySelector(".nav-tab.active");
    const tabName = activeTab ? activeTab.innerText.trim() : "Active Learning";

    const roundVal = document.getElementById("metricRound") ? document.getElementById("metricRound").innerText : "0 / 5";
    const oaVal = document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "52.40%";
    const labeledVal = document.getElementById("metricLabeled") ? document.getElementById("metricLabeled").innerText : "160";

    titleEl.innerText = `💡 Contextual Explanation • ${tabName}`;
    
    let html = ``;

    if (tabName.includes("Active Learning")) {
        const isSampleModalOpen = document.getElementById("sampleModal")?.classList.contains("active");
        if (isSampleModalOpen) {
            html += `
                <div style="background:var(--bg-subtle); padding:10px 14px; border-radius:6px; margin-bottom:10px; border-left:3px solid var(--primary);">
                    <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Sample Inspection</div>
                    <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">
                        This sample was selected for inspection because it was considered informative by the Active Learning strategy.<br><br>
                        <strong>Spectral Signature:</strong> This curve shows the reflectance response across spectral bands.<br>
                        <strong>Class Probabilities:</strong> Shows the classifier confidence distribution across candidate land-cover classes.
                    </div>
                </div>
            `;
        } else {
            const activeMapBtn = document.querySelector(".segmented-btn.active");
            const mapType = activeMapBtn ? activeMapBtn.innerText : "RGB Image";
            
            let mapExplanation = "";
            if (mapType.includes("RGB")) {
                mapExplanation = "<strong>RGB Composite:</strong> Standard hyperspectral band synthesis rendering spectral bands into visible channels.";
            } else if (mapType.includes("Pool")) {
                mapExplanation = "<strong>Label Pool:</strong> Shows which pixels are currently labeled in the training pool versus remaining unlabelled candidate space.";
            } else if (mapType.includes("Prediction")) {
                mapExplanation = "<strong>Prediction Map:</strong> Each pixel represents the class predicted by the classifier.";
            } else {
                mapExplanation = "<strong>Uncertainty Heatmap:</strong> Least Confidence measures how unsure the model is about its best prediction.";
            }

            html += `
                <div style="background:var(--bg-subtle); padding:10px 14px; border-radius:6px; margin-bottom:10px; border-left:3px solid var(--primary);">
                    <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Overall Accuracy</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Overall Accuracy is the percentage of correctly classified evaluation pixels (currently <strong>${oaVal}</strong> across ${labeledVal} labelled samples).</div>
                </div>
                <div style="background:var(--bg-subtle); border-radius:6px; padding:8px 10px; margin-bottom:10px; font-size:12px; color:var(--text-secondary);">
                    ${mapExplanation}
                </div>
                <div style="padding:10px; background:#eff6ff; border-radius:6px; font-size:12px; color:#1e3a8a;">
                    <strong>Next Step:</strong> ${
                        !hasPredicted ? "Click <strong>⚡ Initial Prediction</strong> to generate baseline model performance on the unlabelled scene." :
                        !hasQueried ? `Click <strong>1. Query Samples</strong> to identify candidate pixels via ${currentStrategy}.` :
                        "Assign manual annotations in the table or modal, then click <strong>2. Retrain Classifier</strong> to update the classifier head."
                    }
                </div>
            `;
        }
    } else if (tabName.includes("Cross-Sensor")) {
        const pair = document.getElementById("transferPairSelect") ? document.getElementById("transferPairSelect").value : "Indian Pines -> Pavia University";
        html += `
            <div style="background:var(--bg-subtle); padding:10px 14px; border-radius:6px; margin-bottom:10px; border-left:3px solid var(--primary);">
                <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Cross-Sensor Adaptation</div>
                <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">Different sensors can have different spectral band configurations. Spectral alignment helps bring their representations into a compatible form before target adaptation.</div>
            </div>
            <p style="margin-bottom:8px; font-size:12px;"><strong>Active Pair:</strong> ${pair}</p>
        `;
    } else if (tabName.includes("Experiments")) {
        const expType = document.getElementById("expTypeSelect") ? document.getElementById("expTypeSelect").value : "benchmark";
        let expDetail = "";
        if (expType === "benchmark") {
            expDetail = "Benchmark results compare the framework across evaluated datasets and Active Learning baselines across 5 seeds.";
        } else if (expType === "ablation") {
            expDetail = "Ablation studies show which components actually help. More components do not automatically produce better performance.";
        } else if (expType === "calibration") {
            expDetail = "Calibration evaluates whether the model’s confidence matches the reliability of its predictions (ECE metric).";
        } else if (expType === "cost") {
            expDetail = "This section shows the computational latency of the frozen backbone and lightweight classifier.";
        } else {
            expDetail = "Label efficiency measures accuracy progression per acquired label budget.";
        }
        html += `
            <div style="background:var(--bg-subtle); padding:10px 14px; border-radius:6px; margin-bottom:10px; border-left:3px solid var(--primary);">
                <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Experimental Evidence</div>
                <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">${expDetail}</div>
            </div>
        `;
    } else {
        html += `
            <div style="background:var(--bg-subtle); padding:10px 14px; border-radius:6px; margin-bottom:10px; border-left:3px solid var(--primary);">
                <div style="font-weight:700; color:var(--text-primary); margin-bottom:4px;">Scientific Audit Report</div>
                <div style="font-size:12px; color:var(--text-secondary); line-height:1.5;">The Research Report contains the detailed methodology, experimental evidence, limitations and reproducibility information.</div>
            </div>
        `;
    }

    bodyEl.innerHTML = html;
    modal.classList.add("active");
}

/* =========================================================================
   DATASET LOADING & ACTIVE LEARNING SESSION INIT
   ========================================================================= */
async function loadDataset(dsName, userInitiated = false) {
    isDatasetLoading = true;

    try {
        const res = await fetch("/api/session/select_dataset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_name: dsName, seed: 42 }),
        });
        await res.json();

        // Fetch scene metadata
        const sceneRes = await fetch("/api/scene_data");
        sceneData = await sceneRes.json();
        window.sceneData = sceneData;

        renderSceneMeta();
        renderClassLegend();

        // Ensure 3D viewport active content is immediately visible
        const sceneEmpty = document.getElementById("sceneEmptyState");
        const sceneActive = document.getElementById("sceneActiveContent");
        if (sceneEmpty) sceneEmpty.style.display = "none";
        if (sceneActive) sceneActive.style.display = "block";

        const btnQuery = document.getElementById("btnRunQuery");
        if (btnQuery) {
            btnQuery.disabled = false;
            btnQuery.classList.remove("disabled-stage", "btn-secondary");
            btnQuery.classList.add("btn-primary");
        }

        // Fetch initial baseline predictions so stats & classification data are immediately available
        try {
            await fetchAndRenderPredictionsAndUncertainty();
        } catch (predErr) {
            console.warn("Initial baseline prediction fetch error:", predErr);
        }

        await fetchRoundProgression();
        isDatasetLoading = false;

        // Render 3D scene and update stats panel
        renderScene3D();
        update3DStatsPanel();

        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            if (window.tutorialEngine.currentState === "PREPARE") {
                window.tutorialEngine.renderState();
            }
        }
    } catch (err) {
        isDatasetLoading = false;
        console.error("Error loading dataset:", err);
    }
}

function renderSceneMeta() {
    if (!sceneData) return;
    const [h, w] = sceneData.dimensions;
    const metaEl = document.getElementById("sceneMeta");
    if (metaEl) {
        metaEl.innerHTML = 
            `Dimensions: <strong>${h} &times; ${w}</strong> &bull; Bands: <strong>${sceneData.bands}</strong> &bull; Classes: <strong>${sceneData.num_classes}</strong> &bull; Labeled: <strong>${sceneData.num_labeled}</strong>`;
    }

    const lblEl = document.getElementById("metricLabeled");
    if (lblEl) lblEl.innerText = sceneData.num_labeled;
    const unlblEl = document.getElementById("metricUnlabeled");
    if (unlblEl) unlblEl.innerText = sceneData.num_unlabeled.toLocaleString();
}

function renderClassLegend() {
    if (!sceneData) return;
    const grid = document.getElementById("classLegendGrid");
    if (!grid) return;
    grid.innerHTML = "";

    sceneData.class_names.forEach((name, i) => {
        const color = sceneData.class_colors[i];
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<div class="legend-swatch" style="background:${color};"></div><span>C${i+1}: ${name}</span>`;
        grid.appendChild(item);
    });
}

/* =========================================================================
   CANVAS MAP RENDERING (GROUND TRUTH PRIVACY PROTECTED IN HUMAN MODE)
   ========================================================================= */
function switchMap(mapType) {
    currentMapType = mapType;

    const mapBtns = document.querySelectorAll("#sceneMapControls .segmented-btn");
    mapBtns.forEach(b => b.classList.remove("active"));
    if (mapType === "rgb" && document.getElementById("btnMapRGB")) document.getElementById("btnMapRGB").classList.add("active");
    if (mapType === "pool" && document.getElementById("btnMapPool")) document.getElementById("btnMapPool").classList.add("active");
    if (mapType === "pred" && document.getElementById("btnMapPred")) document.getElementById("btnMapPred").classList.add("active");
    if (mapType === "unc" && document.getElementById("btnMapUnc")) document.getElementById("btnMapUnc").classList.add("active");

    renderSpatialCanvas();
}

async function fetchAndRenderPredictionsAndUncertainty() {
    try {
        const res = await fetch(`/api/predictions_and_uncertainty?uncertainty_method=${currentUncertaintyMethod}`);
        lastPredictionsData = await res.json();
        window.lastPredictionsData = lastPredictionsData;
        renderSpatialCanvas();
    } catch (err) {
        console.error("Error fetching predictions:", err);
    }
}

function renderSpatialCanvas() {
    if (!sceneData) return;
    const canvas = document.getElementById("spatialCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const [height, width] = sceneData.dimensions;
    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    if (currentMapType === "rgb") {
        const rgb = sceneData.rgb_uint8;
        for (let r = 0; r < height; r++) {
            for (let c = 0; c < width; c++) {
                const idx = (r * width + c) * 4;
                const px = rgb[r][c];
                data[idx] = px[0];
                data[idx+1] = px[1];
                data[idx+2] = px[2];
                data[idx+3] = 255;
            }
        }
    } else if (currentMapType === "pool") {
        const mask = sceneData.spatial_mask;
        const gt = sceneData.gt_labels;
        for (let r = 0; r < height; r++) {
            for (let c = 0; c < width; c++) {
                const idx = (r * width + c) * 4;
                const val = mask[r][c];
                if (gt[r][c] === 0) {
                    data[idx] = 241; data[idx+1] = 245; data[idx+2] = 249; data[idx+3] = 255;
                } else if (val === 1) {
                    data[idx] = 37; data[idx+1] = 99; data[idx+2] = 235; data[idx+3] = 255;
                } else {
                    data[idx] = 203; data[idx+1] = 213; data[idx+2] = 225; data[idx+3] = 255;
                }
            }
        }
    } else if (currentMapType === "pred" && lastPredictionsData) {
        const predMap = lastPredictionsData.prediction_map;
        const gt = sceneData.gt_labels;
        for (let r = 0; r < height; r++) {
            for (let c = 0; c < width; c++) {
                const idx = (r * width + c) * 4;
                const pCls = predMap[r][c];
                if (gt[r][c] === 0 || pCls === -1) {
                    data[idx] = 241; data[idx+1] = 245; data[idx+2] = 249; data[idx+3] = 255;
                } else {
                    const hex = sceneData.class_colors[pCls];
                    const [red, green, blue] = hexToRgbArray(hex);
                    data[idx] = red; data[idx+1] = green; data[idx+2] = blue; data[idx+3] = 255;
                }
            }
        }
    } else if (currentMapType === "unc" && lastPredictionsData) {
        const uncMap = lastPredictionsData.uncertainty_map;
        for (let r = 0; r < height; r++) {
            for (let c = 0; c < width; c++) {
                const idx = (r * width + c) * 4;
                const val = uncMap[r][c];
                data[idx] = Math.floor(val * 220);
                data[idx+1] = Math.floor((1 - val) * 100);
                data[idx+2] = Math.floor((1 - val) * 220);
                data[idx+3] = val > 0 ? 255 : 0;
            }
        }
    }

    ctx.putImageData(imgData, 0, 0);

    // Overlay queried points
    if (querySelectionData && querySelectionData.selected_samples) {
        ctx.fillStyle = "#dc2626";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        querySelectionData.selected_samples.forEach(s => {
            const [r, c] = s.coord;
            ctx.beginPath();
            ctx.arc(c + 0.5, r + 0.5, 2.5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });
    }

    // Hover info event with STRICT GT PRIVACY IN HUMAN MODE
    canvas.onmousemove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const c = Math.floor((e.clientX - rect.left) * scaleX);
        const r = Math.floor((e.clientY - rect.top) * scaleY);

        if (r >= 0 && r < height && c >= 0 && c < width) {
            const gtCls = sceneData.gt_labels[r][c];
            const pCls = lastPredictionsData ? lastPredictionsData.prediction_map[r][c] : -1;
            const unc = lastPredictionsData ? lastPredictionsData.uncertainty_map[r][c] : 0.0;
            
            // STRICT GROUND TRUTH PROTECTION IN HUMAN ANNOTATION MODE
            const gtName = (currentMode === "human") 
                ? "HIDDEN (Human Mode)" 
                : (gtCls > 0 ? sceneData.class_names[gtCls-1] : "Background");

            const predName = pCls >= 0 ? sceneData.class_names[pCls] : "N/A";

            const hEl = document.getElementById("hoverInfo");
            if (hEl) {
                hEl.innerText = `Coord: (${r}, ${c}) | True Class: ${gtName} | Predicted: ${predName} | Uncertainty: ${unc.toFixed(3)}`;
            }
        }
    };
}

async function setSessionMode(mode) {
    try {
        const res = await fetch("/api/session/set_mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: mode }),
        });
        const data = await res.json();
        currentMode = data.mode;

        const badge = document.getElementById("modeBadge");
        if (badge) {
            badge.innerText = currentMode === "human" ? "HUMAN ANNOTATION MODE" : "SIMULATED ORACLE / EXPERIMENTAL";
            badge.className = currentMode === "human" ? "badge badge-success" : "badge badge-primary";
        }

        const btnRetrain = document.getElementById("btnSubmitAnnotate");
        if (btnRetrain) {
            btnRetrain.innerText = currentMode === "human" ? "2. Submit Annotations & Retrain" : "2. Retrain Classifier";
        }

        renderQuerySelectionTable();
        renderSpatialCanvas();
    } catch (err) {
        console.error("Error setting session mode:", err);
    }
}

async function setSessionTargetOA(val) {
    try {
        const res = await fetch("/api/session/set_target", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ target_oa: val }),
        });
        const data = await res.json();
        updateTargetBadge(data.target_status);
    } catch (err) {
        console.error("Error setting target OA:", err);
    }
}

function updateTargetBadge(targetStatus) {
    const badge = document.getElementById("targetStatusBadge");
    if (!badge || !targetStatus) return;
    if (targetStatus.target_reached) {
        badge.innerText = `TARGET REACHED (${targetStatus.current_oa.toFixed(1)}% >= ${targetStatus.target_oa}%)`;
        badge.className = "badge badge-success";
    } else {
        badge.innerText = `TARGET NOT REACHED (${targetStatus.current_oa.toFixed(1)}% / ${targetStatus.target_oa}%)`;
        badge.className = "badge badge-warning";
    }
}

/* =========================================================================
   QUERY SELECTION & RANKED SAMPLES TABLE
   ========================================================================= */
async function runQuerySelection() {
    try {
        const kSel = document.getElementById("queryBatchSelect");
        if (kSel) currentQueryBatchSize = parseInt(kSel.value) || 20;

        const res = await fetch("/api/query_selection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ k: currentQueryBatchSize, uncertainty_method: currentStrategy }),
        });
        querySelectionData = await res.json();
        window.querySelectionData = querySelectionData;

        const btnRetrain = document.getElementById("btnSubmitAnnotate");
        if (btnRetrain) {
            btnRetrain.disabled = false;
            btnRetrain.classList.remove("disabled-stage", "btn-secondary");
            btnRetrain.classList.add("btn-success");
        }

        renderQuerySelectionTable();
        renderSpatialCanvas();
        renderScene3D();
        update3DStatsPanel();

        switchMap("unc");
        // Guided State Machine Event
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            const count = (querySelectionData.selected_samples && querySelectionData.selected_samples.length) ? querySelectionData.selected_samples.length : currentQueryBatchSize;
            window.tutorialEngine.sessionMetrics.querySize = count;
            window.tutorialEngine.showResultComment(
                `${count} INFORMATIVE SAMPLES SELECTED`,
                `<p><strong>${count} samples were selected.</strong></p>
                 <p>These samples are candidates for annotation because the selected Active Learning strategy considers them informative.</p>
                 <p>Least Confidence focuses on uncertain predictions, while the diversity component helps reduce redundant selections.</p>`,
                "Scroll down to inspect one of the selected samples.",
                "#spatialCanvas"
            );
        }
    } catch (err) {
        console.error("Error running query selection:", err);
    }
}

function renderQuerySelectionTable() {
    if (!querySelectionData || !querySelectionData.selected_samples) return;
    const tbody = document.querySelector("#querySelectionTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const btnOracle = document.getElementById("btnAutoFillOracle");
    if (btnOracle) {
        btnOracle.style.display = currentMode === "human" ? "inline-block" : "none";
    }

    querySelectionData.selected_samples.forEach((s) => {
        const tr = document.createElement("tr");

        let classSelectHtml = "";
        if (sceneData && sceneData.class_names) {
            const options = sceneData.class_names.map(cName => {
                const isSelected = (s.oracle_class && cName === s.oracle_class) || (!s.oracle_class && cName === s.predicted_class);
                return `<option value="${cName}" ${isSelected ? "selected" : ""}>${cName}</option>`;
            }).join("");
            classSelectHtml = `<select class="select-input user-class-select" data-global-index="${s.global_index}" style="padding:2px 6px; font-size:11px;">${options}</select>`;
        } else {
            classSelectHtml = s.predicted_class;
        }

        const marginDiff = s.margin_diff !== undefined ? s.margin_diff.toFixed(4) : (1.0 - s.least_confidence).toFixed(4);
        const marginUnc = s.margin_uncertainty !== undefined ? s.margin_uncertainty.toFixed(4) : s.least_confidence.toFixed(4);

        tr.innerHTML = `
            <td><strong>#${s.rank}</strong></td>
            <td style="font-family:'JetBrains Mono',monospace;">(${s.coord[0]}, ${s.coord[1]})</td>
            <td><span style="display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px; background:${getClassColor(s.predicted_class)}; color:#ffffff; font-weight:600;">${s.predicted_class}</span></td>
            <td style="font-family:'JetBrains Mono',monospace;"><strong>${marginDiff}</strong></td>
            <td style="font-family:'JetBrains Mono',monospace;"><strong style="color:var(--danger);">${marginUnc}</strong></td>
            <td style="font-size:11px; color:var(--text-secondary); max-width:200px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="${s.selection_reason}">${s.selection_reason}</td>
            <td><button class="btn btn-secondary btn-sm" onclick="inspectSample(${s.global_index}, ${s.rank})">Inspect</button></td>
            <td>${currentMode === "human" ? classSelectHtml : `<span style="font-size:11px; color:var(--text-muted);">Oracle: <strong>${s.oracle_class || 'Verified'}</strong></span>`}</td>
        `;
        tbody.appendChild(tr);
    });
}

function autoFillOracleLabels() {
    if (!querySelectionData || !querySelectionData.selected_samples) return;
    querySelectionData.selected_samples.forEach(s => {
        const sel = document.querySelector(`.user-class-select[data-global-index="${s.global_index}"]`);
        if (sel && s.oracle_class) {
            sel.value = s.oracle_class;
        }
    });
    showToast("Loaded Oracle Ground Truth labels for inspection & retraining.", "success");
}

function getClassColor(clsName) {
    if (!sceneData) return "#2563eb";
    const idx = sceneData.class_names.indexOf(clsName);
    return idx >= 0 ? sceneData.class_colors[idx] : "#2563eb";
}

/* =========================================================================
   SAMPLE INSPECTION MODAL (GROUND TRUTH PRIVACY PROTECTED)
   ========================================================================= */
async function inspectSample(globalIndex, rank = null) {
    try {
        const res = await fetch(`/api/sample_inspection/${globalIndex}`);
        const data = await res.json();

        const displayRank = (data.rank !== undefined && data.rank !== null) ? data.rank : (rank || 1);
        const modalTitle = document.getElementById("modalSampleTitle");
        if (modalTitle) modalTitle.innerText = `Pixel Sample Details • Coord (${data.coord[0]}, ${data.coord[1]}) • Rank #${displayRank}`;

        const modalMeta = document.getElementById("modalSampleMeta");
        if (modalMeta) {
            const gtDisplay = data.mode === "oracle" ? `${data.gt_class} <span style="font-size:10px; color:#2563eb;">(Oracle)</span>` : `HIDDEN (Human Annotation Mode)`;
            modalMeta.innerHTML = `
                <p style="margin-bottom:4px;"><strong>Selection Rank:</strong> #${displayRank}</p>
                <p style="margin-bottom:4px;"><strong>Ground Truth Class:</strong> ${gtDisplay}</p>
                <p style="margin-bottom:4px;"><strong>Predicted Class:</strong> ${data.predicted_class}</p>
                <p style="margin-bottom:4px;"><strong>Least Confidence Score:</strong> ${data.least_confidence.toFixed(3)}</p>
                <p><strong>Diversity Distance Score:</strong> ${data.diversity_score ? data.diversity_score.toFixed(3) : "0.000"}</p>
            `;
        }

        // In-modal annotation dropdown binding
        const cNames = (data.class_names && data.class_names.length) ? data.class_names : ((sceneData && sceneData.class_names) ? sceneData.class_names : []);
        const modalAnnotationSelect = document.getElementById("modalAnnotationSelect");
        if (modalAnnotationSelect && cNames.length) {
            modalAnnotationSelect.innerHTML = cNames.map(cName => {
                const selected = (data.user_annotation === cName || (!data.user_annotation && cName === data.predicted_class)) ? "selected" : "";
                return `<option value="${cName}" ${selected}>${cName}</option>`;
            }).join("");
        }

        const modalApplyBtn = document.getElementById("modalApplyAnnotationBtn");
        if (modalApplyBtn) {
            modalApplyBtn.onclick = () => {
                const chosenCls = modalAnnotationSelect ? modalAnnotationSelect.value : null;
                if (chosenCls) {
                    const tableSelect = document.querySelector(`.user-class-select[data-global-index="${globalIndex}"]`);
                    if (tableSelect) tableSelect.value = chosenCls;
                    showToast(`Updated label for sample #${displayRank} to "${chosenCls}"`, "info");
                    if (window.tutorialEngine && window.tutorialEngine.isActive) {
                        window.tutorialEngine.notifyAction("SAMPLE_ANNOTATED", {
                            mode: currentMode,
                            chosenClass: chosenCls
                        });
                    }
                }
            };
        }

        const reasonBox = document.getElementById("modalReasonBox");
        if (reasonBox) reasonBox.innerText = `Selection Rationale (${data.oracle_badge}): ${data.selection_reason}`;

        // Render 15x15 Patch Canvas
        const pCanvas = document.getElementById("patchCanvas");
        if (pCanvas) {
            const pCtx = pCanvas.getContext("2d");
            pCanvas.width = 15; pCanvas.height = 15;
            const pImg = pCtx.createImageData(15, 15);
            const patchData = data.patch_uint8;
            for (let r = 0; r < 15; r++) {
                for (let c = 0; c < 15; c++) {
                    const idx = (r * 15 + c) * 4;
                    const px = patchData[r][c];
                    pImg.data[idx] = px[0];
                    pImg.data[idx+1] = px[1];
                    pImg.data[idx+2] = px[2];
                    pImg.data[idx+3] = 255;
                }
            }
            pCtx.putImageData(pImg, 0, 0);
        }

        // Render Spectrum Chart (3D / 2D Plotly)
        renderSampleSpectrumChart(data);

        // Render Top Class Probabilities Bar Chart (with actual human-readable class names!)
        const probEl = document.getElementById("sampleProbChart");
        if (probEl && data.probabilities) {
            const pCtx = probEl.getContext("2d");
            if (sampleProbChart) sampleProbChart.destroy();

            const indexedProbs = data.probabilities.map((prob, idx) => ({
                class: cNames[idx] || `Class ${idx + 1}`,
                prob
            }));
            indexedProbs.sort((a, b) => b.prob - a.prob);
            const top4 = indexedProbs.slice(0, 4);

            sampleProbChart = new Chart(pCtx, {
                type: "bar",
                data: {
                    labels: top4.map(item => item.class),
                    datasets: [{
                        label: "Probability",
                        data: top4.map(item => (item.prob * 100).toFixed(1)),
                        backgroundColor: ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
                        borderRadius: 4,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { max: 100, grid: { color: "#e2e8f0" }, ticks: { color: "#64748b", font: { size: 10 }, callback: v => v + '%' } },
                        y: { grid: { display: false }, ticks: { color: "#0f172a", font: { size: 10, weight: '600' } } }
                    }
                }
            });
        }

        // Developer verification audit
        logHsiSampleDebug(data);

        const modalBackdrop = document.getElementById("sampleModal");
        if (modalBackdrop) modalBackdrop.classList.add("active");

        // Modal Close Listener Initialization (once)
        const closeBtn = document.getElementById("modalCloseBtn");
        if (closeBtn) {
            closeBtn.onclick = () => {
                if (modalBackdrop) modalBackdrop.classList.remove("active");
                if (window.tutorialEngine && window.tutorialEngine.isActive &&
                    (window.tutorialEngine.currentState === "INSPECT_MODAL" || window.tutorialEngine.currentState === "ANNOTATE")) {
                    window.tutorialEngine.advanceTo("ANNOTATE_RESULT");
                }
            };
        }
        if (modalBackdrop) {
            modalBackdrop.onclick = (e) => {
                if (e.target === modalBackdrop) {
                    modalBackdrop.classList.remove("active");
                    if (window.tutorialEngine && window.tutorialEngine.isActive &&
                        (window.tutorialEngine.currentState === "INSPECT_MODAL" || window.tutorialEngine.currentState === "ANNOTATE")) {
                        window.tutorialEngine.advanceTo("ANNOTATE_RESULT");
                    }
                }
            };
        }

        // Notify tutorial engine that a sample was inspected
        if (window.tutorialEngine) {
            window.tutorialEngine.notifyAction("SAMPLE_INSPECTED");
        }
    } catch (err) {
        console.error("Error inspecting sample:", err);
        showToast("Error inspecting sample", "warning");
    }
}

/* =========================================================================
   ANNOTATION & RETRAINING (HUMAN & ORACLE MODES)
   ========================================================================= */
async function annotateAndRetrain() {
    try {
        const prevLabeled = sceneData ? sceneData.num_labeled : 160;
        const prevOaText = document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "52.40%";
        const prevOaVal = parseFloat(prevOaText) || 52.40;
        const prevF1 = document.getElementById("metricF1") ? document.getElementById("metricF1").innerText : "0.4520";

        let endpoint = "/api/annotate_and_retrain";
        let bodyPayload = { epochs: 25, lr: 0.0005 };

        if (currentMode === "human") {
            endpoint = "/api/annotate";
            const selects = document.querySelectorAll(".user-class-select");
            const annotations = [];
            selects.forEach(sel => {
                const gIdx = parseInt(sel.getAttribute("data-global-index"));
                const selCls = sel.value;
                annotations.push({ global_index: gIdx, selected_class: selCls });
            });
            if (annotations.length === 0) {
                alert("Please click '1. Query Samples' first to generate queries to annotate.");
                return;
            }
            bodyPayload = { annotations: annotations, epochs: 25, lr: 0.0005 };
        }

        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(bodyPayload),
        });
        const data = await res.json();

        // Update metrics display
        const m = data.metrics;
        const roundEl = document.getElementById("metricRound");
        if (roundEl) roundEl.innerText = `${data.current_round} / 5`;
        const lblEl = document.getElementById("metricLabeled");
        if (lblEl) lblEl.innerText = data.num_labeled;
        const unlblEl = document.getElementById("metricUnlabeled");
        if (unlblEl) unlblEl.innerText = data.num_unlabeled.toLocaleString();

        if (data.num_pseudo !== undefined) {
            const pseudoMetricEl = document.getElementById("metricPseudo");
            const pseudoDotEl = document.getElementById("pseudoStatusDot");
            const pseudoLabelEl = document.getElementById("pseudoStatusLabel");
            if (pseudoMetricEl) pseudoMetricEl.innerText = data.num_pseudo;
            if (data.num_pseudo > 0) {
                if (pseudoDotEl) pseudoDotEl.style.display = "inline";
                if (pseudoLabelEl) pseudoLabelEl.style.display = "inline";
            }
        }

        if (data.target_status) {
            updateTargetBadge(data.target_status);
        }

        const newOaVal = m.OverallAccuracy * 100;
        const newOA = `${newOaVal.toFixed(2)}%`;
        const newF1 = m.F1Score.toFixed(4);

        const oaEl = document.getElementById("metricOA");
        if (oaEl) oaEl.innerText = newOA;
        const f1El = document.getElementById("metricF1");
        if (f1El) f1El.innerText = newF1;

        // Compact Before / After comparison
        if (document.getElementById("baLabels")) document.getElementById("baLabels").innerText = `${prevLabeled} → ${data.num_labeled}`;
        if (document.getElementById("baOA")) document.getElementById("baOA").innerText = `${prevOaText} → ${newOA}`;
        if (document.getElementById("baF1")) document.getElementById("baF1").innerText = `${prevF1} → ${newF1}`;

        // Fetch predictions to update map
        await fetchAndRenderPredictionsAndUncertainty();

        // Clear query table
        querySelectionData = null;
        const tbody = document.querySelector("#querySelectionTable tbody");
        if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">Retraining complete! Click "1. Query Samples" for next round.</td></tr>`;

        // Refresh progression table & curves
        fetchRoundProgression();

        // Mark retrained state and ensure 3D prediction layer is active
        hasRetrained = true;
        show3DPredictions = true;
        const togglePred = document.getElementById("toggle3dPredictions");
        if (togglePred) togglePred.checked = true;

        // Update 3D scene, stats, and 2D canvas
        renderSpatialCanvas();
        renderScene3D();
        update3DStatsPanel();

        const badgeEl = document.getElementById("scene3dSamplesBadge");
        if (badgeEl && lastPredictionsData && lastPredictionsData.statistics) {
            badgeEl.innerHTML = `Post-Annotation 3D Classification Map &bull; Round ${lastPredictionsData.statistics.round} &bull; ${lastPredictionsData.statistics.predicted_pixels.toLocaleString()} Pixels Classified`;
        }
        const infoEl = document.getElementById("scene3dSelectedInfo");
        if (infoEl) infoEl.innerHTML = `<span style="color:var(--success); font-weight:600; margin-right:6px;">✓</span> Retraining complete. 3D surface updated with model predictions across all eligible pixels.`;

        switchMap("pred");
        // Guided State Machine Event
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("RETRAIN_DONE", {
                old_oa: prevOaText.replace("%", ""),
                new_oa: newOA.replace("%", ""),
                num_labeled: data.num_labeled
            });
        }
    } catch (err) {
        console.error("Error during annotation & retraining:", err);
    }
}

async function calculateLabelCost() {
    try {
        const costInput = document.getElementById("costPerLabelInput");
        const costPerLabel = parseFloat(costInput ? costInput.value : 5.0) || 5.0;
        const dsName = window.currentDataset || "Indian Pines";

        const res = await fetch("/api/calculate_cost", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cost_per_label: costPerLabel, dataset_name: dsName }),
        });
        const data = await res.json();

        const randCost = data.random ? data.random.cost : 900 * costPerLabel;
        const randLabels = data.random ? data.random.labels : 900;
        const marginCost = data.margin ? data.margin.cost : 700 * costPerLabel;
        const marginLabels = data.margin ? data.margin.labels : 700;
        const costSaved = data.savings ? data.savings.cost_saved : (randCost - marginCost);
        const pctSaved = data.savings ? data.savings.percentage_workload_reduction : 22.2;
        const labelsSaved = data.savings ? data.savings.labels_saved : (randLabels - marginLabels);
        const targetOa = data.target_oa_ref || "66.0%";

        if (document.getElementById("costRandom")) document.getElementById("costRandom").innerText = `$${Math.round(randCost).toLocaleString()}`;
        if (document.getElementById("labelsRandom")) document.getElementById("labelsRandom").innerText = `${randLabels} samples (at ${targetOa} OA)`;

        if (document.getElementById("costLCFD")) document.getElementById("costLCFD").innerText = `$${Math.round(marginCost).toLocaleString()}`;
        if (document.getElementById("labelsLCFD")) document.getElementById("labelsLCFD").innerText = `${marginLabels} samples (Pure Margin)`;

        if (document.getElementById("labelsSaved")) {
            const sign = labelsSaved > 0 ? "+" : "";
            const color = labelsSaved > 0 ? "var(--success)" : (labelsSaved < 0 ? "var(--danger)" : "var(--text-secondary)");
            document.getElementById("labelsSaved").innerHTML = `<span style="color:${color};">${sign}${labelsSaved} samples</span>`;
        }

        if (document.getElementById("costSaved")) {
            const sign = costSaved > 0 ? "+" : "";
            const color = costSaved > 0 ? "var(--success)" : (costSaved < 0 ? "var(--danger)" : "var(--text-secondary)");
            document.getElementById("costSaved").innerHTML = `<span style="color:${color};">${sign}$${Math.round(costSaved).toLocaleString()} (${sign}${pctSaved}%)</span>`;
        }
    } catch (err) {
        console.error("Error calculating label cost:", err);
    }
}

/* =========================================================================
   ACTIVE LEARNING PROGRESSION
   ========================================================================= */
async function fetchRoundProgression() {
    try {
        const res = await fetch("/api/round_progression");
        const data = await res.json();
        const history = data.history;

        const pSel = document.getElementById("pseudoToggleSelect");
        const pDot = document.getElementById("pseudoStatusDot");
        const pLabel = document.getElementById("pseudoStatusLabel");
        const pMetric = document.getElementById("metricPseudo");

        if (data.use_pseudo_labeling) {
            if (pSel) pSel.value = "on";
            if (pDot) pDot.style.display = "inline";
            if (pLabel) pLabel.style.display = "inline";
            if (pMetric && data.last_pseudo_stats) {
                pMetric.innerText = data.last_pseudo_stats.count || 0;
            }
        } else {
            if (pSel) pSel.value = "off";
            if (pDot) pDot.style.display = "none";
            if (pLabel) pLabel.style.display = "none";
        }

        renderProgressionChart(history);
    } catch (err) {
        console.error("Error fetching progression:", err);
    }
}

function renderProgressionChart(history) {
    const chartEl = document.getElementById("progressionChart");
    showGraphGeneratingAnimation(chartEl.parentElement || chartEl, "Generating Active Learning Curves...");
    if (!chartEl) return;

    // Cache history
    if (history && history.length > 0) {
        lastProgressionHistory = history;
    } else if (!lastProgressionHistory) {
        lastProgressionHistory = [
            { round: 0, num_labeled: (sceneData && sceneData.num_labeled) || 160, metrics: { OverallAccuracy: 0.4532, F1Score: 0.4307 } }
        ];
    }
    const dataToRender = lastProgressionHistory;

    // Ensure active progression container is visible
    const progEmpty = document.getElementById("progressionEmptyState");
    const progActive = document.getElementById("progressionActiveContent");
    if (progEmpty) progEmpty.style.display = "none";
    if (progActive) progActive.style.display = "block";

    // Update metric summary box with latest round values
    const latest = dataToRender[dataToRender.length - 1];
    if (latest && latest.metrics) {
        const metricOA = document.getElementById("metricOA");
        const metricF1 = document.getElementById("metricF1");
        if (metricOA) metricOA.innerText = `${(latest.metrics.OverallAccuracy * 100).toFixed(2)}%`;
        if (metricF1) metricF1.innerText = latest.metrics.F1Score.toFixed(4);
    }

    const mode = chartModes.progression || '3d';
    const is3D = (mode === '3d');

    const labels = dataToRender.map(r => `R${r.round} (${r.num_labeled})`);
    const xVals = dataToRender.map(r => r.num_labeled);
    const oas = dataToRender.map(r => +(r.metrics.OverallAccuracy * 100).toFixed(2));
    const f1s = dataToRender.map(r => +(r.metrics.F1Score * 100).toFixed(2));
    const rounds = dataToRender.map(r => r.round);

    if (is3D && window.Plotly) {
        // 3D Active Learning Trajectory
        const traceOA = {
            type: 'scatter3d',
            mode: 'lines+markers+text',
            name: 'Overall Accuracy (%)',
            x: xVals,
            y: Array(xVals.length).fill(0),
            z: oas,
            text: rounds.map(r => `R${r}`),
            textposition: 'top center',
            textfont: { family: 'Inter, sans-serif', size: 10, color: '#1e3a8a' },
            line: { color: '#2563eb', width: 6 },
            marker: { size: 6, color: '#1d4ed8', symbol: 'circle', opacity: 0.95 },
            hovertemplate: '<b>Overall Accuracy</b><br>Round: %{text}<br>Labels: %{x}<br>OA: %{z:.2f}%<extra></extra>'
        };

        const traceF1 = {
            type: 'scatter3d',
            mode: 'lines+markers',
            name: 'Macro F1 (x100)',
            x: xVals,
            y: Array(xVals.length).fill(1),
            z: f1s,
            line: { color: '#10b981', width: 5, dash: 'dash' },
            marker: { size: 5, color: '#059669', symbol: 'diamond' },
            hovertemplate: '<b>Macro F1</b><br>Labels: %{x}<br>Score: %{z:.2f}%<extra></extra>'
        };

        const minX = Math.min(...xVals) * 0.95;
        const maxX = Math.max(...xVals) * 1.05 + 10;
        const targetVal = (typeof targetOA !== 'undefined' ? targetOA : 70.0);

        const traceTarget = {
            type: 'mesh3d',
            name: `Target OA (${targetVal}%)`,
            x: [minX, maxX, maxX, minX],
            y: [-0.3, -0.3, 1.3, 1.3],
            z: [targetVal, targetVal, targetVal, targetVal],
            i: [0, 0],
            j: [1, 2],
            k: [2, 3],
            opacity: 0.18,
            color: '#f59e0b',
            hoverinfo: 'name'
        };

        const traces = [traceOA, traceF1, traceTarget];

        // Add 3D ground drop stems
        dataToRender.forEach((r, idx) => {
            traces.push({
                type: 'scatter3d',
                mode: 'lines',
                showlegend: false,
                x: [xVals[idx], xVals[idx]],
                y: [0, 0],
                z: [0, oas[idx]],
                line: { color: 'rgba(37, 99, 235, 0.35)', width: 2, dash: 'dot' },
                hoverinfo: 'skip'
            });
        });

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: {
                    title: { text: 'Labeled Samples', font: { size: 10, color: '#64748b' } },
                    gridcolor: '#e2e8f0',
                    zerolinecolor: '#cbd5e1',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                yaxis: {
                    title: { text: 'Metric Track', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 1],
                    ticktext: ['Accuracy', 'Macro F1'],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                zaxis: {
                    title: { text: 'Score (%)', font: { size: 10, color: '#64748b' } },
                    range: [0, 100],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                camera: {
                    eye: { x: 1.5, y: -1.7, z: 1.1 },
                    center: { x: 0, y: 0, z: -0.1 }
                }
            },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255, 255, 255, 0.85)',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
                font: { size: 10 }
            }
        };

        const config = { responsive: true, displaylogo: false };
        Plotly.react(chartEl, traces, layout, config);
    setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 200);
    } else if (window.Plotly) {
        // 2D Progression View via Plotly
        const traceOA = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Overall Accuracy (%)',
            x: labels,
            y: oas,
            line: { color: '#2563eb', width: 3 },
            marker: { size: 7, color: '#1d4ed8' },
            fill: 'tozeroy',
            fillcolor: 'rgba(37, 99, 235, 0.08)'
        };

        const traceF1 = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Macro F1 (x100)',
            x: labels,
            y: f1s,
            line: { color: '#10b981', width: 2, dash: 'dash' },
            marker: { size: 6, color: '#059669' }
        };

        const targetVal = (typeof targetOA !== 'undefined' ? targetOA : 70.0);
        const traceTarget = {
            type: 'scatter',
            mode: 'lines',
            name: `Target (${targetVal}%)`,
            x: labels,
            y: Array(labels.length).fill(targetVal),
            line: { color: '#f59e0b', width: 2, dash: 'dot' }
        };

        const layout = {
            margin: { l: 45, r: 20, b: 35, t: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            yaxis: { gridcolor: '#f1f5f9', title: 'Accuracy (%)', range: [0, 100], tickfont: { size: 10, color: '#64748b' } },
            legend: { x: 0.02, y: 0.98, font: { size: 10 } }
        };

        Plotly.react(chartEl, [traceOA, traceF1, traceTarget], layout, { responsive: true, displaylogo: false });
    }
}

/* =========================================================================
   CROSS-SENSOR DEMONSTRATION & 1D SPECTRAL RESAMPLER
   ========================================================================= */
let _crossSensorCache = {};

async function loadCrossSensorPair(pairKey, userInitiated = false) {
    window._crossSensorLoaded = true;
    try {
        if (!_crossSensorCache[pairKey]) {
            const res = await fetch(`/api/cross_sensor_data?pair_key=${encodeURIComponent(pairKey)}`);
            _crossSensorCache[pairKey] = await res.json();
        }
        const data = _crossSensorCache[pairKey];
        const summary = data.summary || {};
        const pInfo = summary.pair_info || summary.metadata || {};
        const mSum = summary.metrics_summary || {};
        const zs = summary.zero_shot_metrics || {};

        if (document.getElementById("csSource")) {
            document.getElementById("csSource").innerText = mSum.source_sensor || pInfo.source_sensor || "AVIRIS";
        }
        if (document.getElementById("csSourceSub")) {
            const sBands = mSum.source_bands || pInfo.source_bands || 200;
            const sGsd = (mSum.source_resolution_m || pInfo.source_resolution_m || 20.0).toFixed(1);
            document.getElementById("csSourceSub").innerText = `${mSum.source_dataset || pInfo.source_dataset || "Source"} (${sBands} bands, ${sGsd}m GSD)`;
        }
        if (document.getElementById("csTarget")) {
            document.getElementById("csTarget").innerText = mSum.target_sensor || pInfo.target_sensor || "AVIRIS";
        }
        if (document.getElementById("csTargetSub")) {
            const tBands = mSum.target_bands || pInfo.target_bands || 204;
            const tGsd = (mSum.target_resolution_m || pInfo.target_resolution_m || 3.7).toFixed(1);
            document.getElementById("csTargetSub").innerText = `${mSum.target_dataset || pInfo.target_dataset || "Target"} (${tBands} bands, ${tGsd}m GSD)`;
        }

        if (document.getElementById("csZeroShot")) {
            if (zs && zs.oa != null && !isNaN(zs.oa)) {
                document.getElementById("csZeroShot").innerText = `${Number(zs.oa).toFixed(2)}%`;
            } else {
                document.getElementById("csZeroShot").innerText = "N/A";
            }
        }
        if (document.getElementById("csZeroShotSub")) {
            document.getElementById("csZeroShotSub").innerText = zs.status || "Disjoint semantic classes";
        }

        if (document.getElementById("csFewShot")) {
            if (mSum.transferred_oa != null && !isNaN(mSum.transferred_oa)) {
                document.getElementById("csFewShot").innerText = `${Number(mSum.transferred_oa).toFixed(2)}%`;
            } else {
                document.getElementById("csFewShot").innerText = "N/A";
            }
        }
        if (document.getElementById("csDeltaSub")) {
            if (mSum.transfer_delta_oa != null && !isNaN(mSum.transfer_delta_oa)) {
                const delta = Number(mSum.transfer_delta_oa);
                const sign = delta > 0 ? "+" : "";
                const deltaColor = delta > 0 ? "var(--success)" : (delta < 0 ? "var(--danger)" : "var(--text-secondary)");
                document.getElementById("csDeltaSub").innerHTML = `&Delta;OA = <strong style="color:${deltaColor}">${sign}${delta.toFixed(2)}%</strong> (${mSum.outcome || 'Neutral'})`;
            } else {
                document.getElementById("csDeltaSub").innerText = "N/A (Baseline unavailable)";
            }
        }

        const catBadge = document.getElementById("csCategoryBadge");
        if (catBadge) {
            const transferType = mSum.transfer_type || pInfo.transfer_type || pInfo.pair_type || "Transfer";
            catBadge.innerText = transferType;
            if (transferType.toLowerCase().includes("genuine")) {
                catBadge.className = "badge badge-danger";
            } else {
                catBadge.className = "badge badge-success";
            }
        }

        window._lastCsMeta = pInfo;
        renderCrossSensorCurves(summary.four_configs || summary.learning_curves, pInfo);
        render1DSpectralResamplerChart(data.resampler_data);
        renderCrossSensorProgressionTable(summary.round_progression);
        updateCrossSensorExplanations(pInfo, mSum, data.resampler_data);

        setTimeout(() => {
            const el1 = document.getElementById("csCurvesChart");
            if (el1 && window.Plotly && el1.offsetHeight > 0) Plotly.Plots.resize(el1);
            const el2 = document.getElementById("resamplerChart");
            if (el2 && window.Plotly && el2.offsetHeight > 0) Plotly.Plots.resize(el2);
        }, 80);

        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("CROSS_SENSOR_DONE", {
                pairKey,
                fewShotVal: `${(mSum.transferred_oa || 0).toFixed(2)}%`,
                sourceSensor: pInfo.source_sensor,
                targetSensor: pInfo.target_sensor
            });
        }
    } catch (err) {
        console.error("Error loading cross-sensor data:", err);
    }
}

const loadCrossSensorData = loadCrossSensorPair;

function renderCrossSensorCurves(fourConfigs, pInfo) {
    const chartEl = document.getElementById("csCurvesChart");
    showGraphGeneratingAnimation(chartEl.parentElement || chartEl, "Synthesizing Cross-Sensor Adaptation Curves...");
    if (!chartEl || !window.Plotly || !fourConfigs) return;
    lastCsCurvesData = fourConfigs;
    if (pInfo) window._lastCsMeta = pInfo;
    const meta = pInfo || window._lastCsMeta || {};

    const srcDs = meta.source_dataset || "Source";
    const tgtDs = meta.target_dataset || "Target";
    const srcSensor = meta.source_sensor || "AVIRIS";
    const tgtSensor = meta.target_sensor || "AVIRIS";

    const is3D = (chartModes.csCurves === "3d");

    const seriesDefs = [
        {
            key: "target_margin",
            altKeys: ["target_only_al", "target_margin_al"],
            name: "Target-only Pure Margin (Target Initialized, 5 Seeds)",
            color: "#2563eb",
            dash: "solid",
            width: 3.2,
            symbol: "diamond",
            size: 8
        },
        {
            key: "source_margin",
            altKeys: ["transferred_al", "source_margin_al"],
            name: "Transferred Pure Margin Adaptation (Source Pre-trained, 5 Seeds)",
            color: "#10b981",
            dash: "solid",
            width: 3.0,
            symbol: "square",
            size: 7
        },
        {
            key: "target_random",
            altKeys: ["target_only_random"],
            name: "Target-only Random Baseline (Target Initialized, 5 Seeds)",
            color: "#64748b",
            dash: "dot",
            width: 2.2,
            symbol: "circle",
            size: 6
        },
        {
            key: "source_random",
            altKeys: ["transferred_frozen", "source_random_al"],
            name: "Transferred Random Adaptation (Source Pre-trained, 5 Seeds)",
            color: "#f59e0b",
            dash: "dash",
            width: 2.2,
            symbol: "triangle-up",
            size: 6
        }
    ];

    function extractPoints(def) {
        let entry = fourConfigs[def.key];
        if (!entry) {
            for (const alt of def.altKeys) {
                if (fourConfigs[alt]) {
                    entry = fourConfigs[alt];
                    break;
                }
            }
        }
        if (!entry) return [];
        const rawPts = Array.isArray(entry) ? entry : (entry.points || []);
        return rawPts.map(p => {
            const budget = p.budget !== undefined ? p.budget : (p.x !== undefined ? p.x : 0);
            const oa = p.oa_mean !== undefined ? p.oa_mean : (p.oa !== undefined ? p.oa : (p.y !== undefined ? p.y : 0));
            const std = p.oa_std !== undefined ? p.oa_std : 0;
            return { budget, oa, std };
        });
    }

    if (is3D) {
        const traces = seriesDefs.map((def, idx) => {
            const pts = extractPoints(def);
            return {
                type: "scatter3d",
                mode: "lines+markers",
                name: def.name,
                x: pts.map(p => p.budget),
                y: Array(pts.length).fill(idx),
                z: pts.map(p => p.oa),
                line: { color: def.color, width: 5, dash: def.dash },
                marker: { size: def.size, color: def.color, symbol: def.symbol },
                hovertemplate: `<b>${def.name}</b><br>Budget N: %{x}<br>OA: %{z:.2f}%<extra></extra>`
            };
        });

        const layout = {
            title: {
                text: `<b>${srcDs} → ${tgtDs}</b> (${srcSensor} → ${tgtSensor})`,
                font: { size: 12, color: "#0f172a" },
                x: 0.05,
                y: 0.98
            },
            margin: { l: 0, r: 0, b: 0, t: 30 },
            paper_bgcolor: "rgba(0,0,0,0)",
            scene: {
                xaxis: { title: "Budget (N)", gridcolor: "#e2e8f0", backgroundcolor: "rgba(248, 250, 252, 0.5)", showbackground: true },
                yaxis: {
                    title: "Strategy",
                    tickvals: [0, 1, 2, 3],
                    ticktext: ["Target Margin", "Transferred Margin", "Target Random", "Transferred Random"],
                    gridcolor: "#e2e8f0",
                    backgroundcolor: "rgba(248, 250, 252, 0.5)",
                    showbackground: true
                },
                zaxis: { title: "Overall Accuracy (%)", gridcolor: "#e2e8f0", backgroundcolor: "rgba(248, 250, 252, 0.5)", showbackground: true },
                camera: { eye: { x: 1.6, y: -1.7, z: 1.2 } }
            },
            legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)', font: { size: 10 } }
        };
        Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
    setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 200);
    } else {
        const traces = seriesDefs.map(def => {
            const pts = extractPoints(def);
            return {
                type: "scatter",
                mode: "lines+markers",
                name: def.name,
                x: pts.map(p => p.budget),
                y: pts.map(p => p.oa),
                error_y: {
                    type: "data",
                    array: pts.map(p => p.std),
                    visible: true,
                    color: def.color,
                    thickness: 1,
                    width: 3
                },
                line: { color: def.color, width: def.width, dash: def.dash },
                marker: { size: def.size, symbol: def.symbol, color: def.color },
                hovertemplate: `<b>${def.name}</b><br>Budget: N=%{x}<br>OA: %{y:.2f}% &plusmn; %{error_y.array:.2f}%<extra></extra>`
            };
        });

        const layout = {
            title: {
                text: `<b>${srcDs} → ${tgtDs}</b> <span style="font-size:11px; color:#64748b; font-weight:normal;">(${srcSensor} → ${tgtSensor} | Protocol A, Seeds 42–46)</span>`,
                font: { size: 12, color: "#0f172a" },
                x: 0.05,
                y: 0.98
            },
            margin: { l: 55, r: 20, b: 40, t: 35 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: {
                title: "Target Labeled Budget (N)",
                gridcolor: "#f1f5f9",
                tickvals: [100, 150, 200, 250, 300, 500],
                tickfont: { size: 10, color: "#64748b" }
            },
            yaxis: {
                title: "Overall Accuracy (OA %)",
                gridcolor: "#f1f5f9",
                tickfont: { size: 10, color: "#64748b" }
            },
            legend: {
                x: 0.02,
                y: 0.03,
                bgcolor: "rgba(255, 255, 255, 0.92)",
                bordercolor: "#e2e8f0",
                borderwidth: 1,
                font: { size: 9.5 }
            }
        };
        Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
        setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 250);
    }
}

function renderCrossSensorProgressionTable(prog) {
    const tbody = document.querySelector("#csProgressionTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!prog || !Array.isArray(prog) || prog.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:16px; color:var(--text-muted);">No progression data available for this transfer pair.</td></tr>`;
        return;
    }

    prog.forEach(r => {
        const tr = document.createElement("tr");
        const b = r.budget !== undefined ? r.budget : "N/A";

        const toRand = (r.target_only_random_oa !== undefined && r.target_only_random_oa !== null) ? `${Number(r.target_only_random_oa).toFixed(2)}%` : "N/A";
        const toMargin = (r.target_only_margin_oa !== undefined && r.target_only_margin_oa !== null) ? `<strong>${Number(r.target_only_margin_oa).toFixed(2)}%</strong>` : "N/A";
        const trRand = (r.transferred_random_oa !== undefined && r.transferred_random_oa !== null) ? `${Number(r.transferred_random_oa).toFixed(2)}%` : "N/A";
        const trMargin = (r.transferred_margin_oa !== undefined && r.transferred_margin_oa !== null) ? `<strong>${Number(r.transferred_margin_oa).toFixed(2)}%</strong>` : "N/A";

        let deltaStr = "N/A";
        let delta = r.transfer_delta;
        if (delta === undefined || delta === null || isNaN(delta)) {
            if (r.transferred_margin_oa !== undefined && r.target_only_margin_oa !== undefined) {
                delta = r.transferred_margin_oa - r.target_only_margin_oa;
            }
        }
        if (delta !== undefined && delta !== null && !isNaN(delta)) {
            const sign = delta > 0 ? "+" : "";
            const color = delta > 0 ? "var(--success)" : (delta < 0 ? "var(--danger)" : "var(--text-secondary)");
            deltaStr = `<strong style="color:${color};">${sign}${Number(delta).toFixed(2)}%</strong>`;
        }

        let pValStr = "N/A";
        if (r.p_value !== undefined && r.p_value !== null && !isNaN(r.p_value)) {
            pValStr = r.p_value < 0.001 ? "<0.001" : Number(r.p_value).toFixed(4);
        }

        let outcomeBadge = `<span class="badge badge-secondary">Neutral</span>`;
        if (r.outcome) {
            if (r.outcome.toLowerCase().includes("positive")) {
                outcomeBadge = `<span class="badge badge-success">Positive</span>`;
            } else if (r.outcome.toLowerCase().includes("negative")) {
                outcomeBadge = `<span class="badge badge-danger">Negative</span>`;
            } else {
                outcomeBadge = `<span class="badge badge-secondary">Neutral (p &gt; 0.05)</span>`;
            }
        } else if (delta !== undefined && delta !== null && !isNaN(delta)) {
            outcomeBadge = delta > 0 ? `<span class="badge badge-success">Positive</span>` : `<span class="badge badge-danger">Negative</span>`;
        }

        tr.innerHTML = `
            <td><strong>N=${b}</strong></td>
            <td>${toRand}</td>
            <td>${toMargin}</td>
            <td>${trRand}</td>
            <td>${trMargin}</td>
            <td>${deltaStr}</td>
            <td>${pValStr}</td>
            <td>${outcomeBadge}</td>
        `;
        tbody.appendChild(tr);
    });
}

function render1DSpectralResamplerChart(resamplerData) {
    const chartEl = document.getElementById("resamplerChart");
    showGraphGeneratingAnimation(chartEl.parentElement || chartEl, "Calculating Continuous Spectral Alignment...");
    if (!chartEl || !window.Plotly || !resamplerData) return;
    lastResamplerData = resamplerData;

    const mode = chartModes.resampler || '2d';
    const is3D = (mode === '3d');

    const waves = resamplerData.master_wavelengths || resamplerData.target_wavelengths || [];
    const srcProfile = resamplerData.source_instrument_profile || resamplerData.resampled_spectrum || [];
    const tgtProfile = resamplerData.target_instrument_profile || resamplerData.target_spectrum || [];
    const isGenuine = resamplerData.is_genuine_cross_sensor === true;
    const srcSensor = resamplerData.source_sensor || "AVIRIS";
    const tgtSensor = resamplerData.target_sensor || "ROSIS";

    const N = Math.min(waves.length, srcProfile.length, tgtProfile.length);
    const subWaves = waves.slice(0, N);
    const subSrc = srcProfile.slice(0, N);
    const subTgt = tgtProfile.slice(0, N);

    if (is3D) {
        const traceAVIRIS = {
            type: 'scatter3d',
            mode: 'lines',
            name: `AVIRIS Sensor Coverage (400–2500 nm, 224 Bands)`,
            x: subWaves,
            y: Array(N).fill(0),
            z: srcSensor === "AVIRIS" ? subSrc : subTgt,
            line: { color: '#2563eb', width: 6 },
            hovertemplate: '<b>AVIRIS Optics</b><br>Wavelength: %{x} nm<br>Normalized Sensitivity: %{z:.2f}<extra></extra>'
        };

        const traceOverlap = {
            type: 'scatter3d',
            mode: 'lines',
            name: `Shared VNIR Overlap Window (430–860 nm)`,
            x: subWaves,
            y: Array(N).fill(1),
            z: subWaves.map(w => (w >= 430 && w <= 860) ? 1.0 : 0.0),
            line: { color: '#10b981', width: 5, dash: 'dot' },
            hovertemplate: '<b>Shared VNIR Overlap</b><br>Wavelength: %{x} nm<extra></extra>'
        };

        const traceROSIS = {
            type: 'scatter3d',
            mode: 'lines',
            name: `ROSIS Sensor Coverage (430–860 nm, VNIR Only)`,
            x: subWaves,
            y: Array(N).fill(2),
            z: tgtSensor === "ROSIS" ? subTgt : subSrc,
            line: { color: '#ef4444', width: 6 },
            hovertemplate: '<b>ROSIS Optics</b><br>Wavelength: %{x} nm<br>Normalized Sensitivity: %{z:.2f}<extra></extra>'
        };

        const layout = {
            title: {
                text: `<b>Optical Compatibility Specs: ${srcSensor} vs ${tgtSensor}</b>`,
                font: { size: 12.5, color: "#0f172a" },
                x: 0.05,
                y: 0.98
            },
            margin: { l: 0, r: 0, b: 0, t: 30 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: {
                    title: { text: 'Wavelength (nm)', font: { size: 10, color: '#64748b' } },
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                yaxis: {
                    title: { text: 'Instrument Track', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 1, 2],
                    ticktext: ['AVIRIS (VNIR-SWIR)', 'Shared VNIR Overlap', 'ROSIS (VNIR Only)'],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                zaxis: {
                    title: { text: 'Bandpass Sensitivity', font: { size: 10, color: '#64748b' } },
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                camera: {
                    eye: { x: 1.6, y: -1.6, z: 1.2 },
                    center: { x: 0, y: 0, z: -0.1 }
                }
            },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255, 255, 255, 0.85)',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
                font: { size: 9.5 }
            }
        };

        Plotly.react(chartEl, [traceAVIRIS, traceOverlap, traceROSIS], layout, { responsive: true, displaylogo: false });
    setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 200);
    } else {
        // 2D Instrument Bandpass Specifications & Coverage Disparity
        const traceAVIRIS = {
            type: 'scatter',
            mode: 'lines',
            name: `${srcSensor === 'AVIRIS' ? resamplerData.source_dataset : resamplerData.target_dataset} AVIRIS Optics (400–2500 nm, VNIR-SWIR)`,
            x: subWaves,
            y: srcSensor === "AVIRIS" ? subSrc : subTgt,
            line: { color: '#2563eb', width: 2.8 },
            hovertemplate: 'AVIRIS Active Bandpass: %{x} nm<extra></extra>'
        };

        const traceTargetSensor = {
            type: 'scatter',
            mode: 'lines',
            name: isGenuine
                ? `ROSIS Instrument Optics (430–860 nm, 115 Bands, VNIR Only)`
                : `${resamplerData.target_dataset} AVIRIS Optics (400–2500 nm, 204 Bands)`,
            x: subWaves,
            y: tgtSensor === "ROSIS" ? subTgt : (isGenuine ? subSrc : subTgt.map(v => v * 0.96)),
            line: { color: '#ef4444', width: 2.5, dash: isGenuine ? 'solid' : 'dash' },
            hovertemplate: `${tgtSensor} Active Bandpass: %{x} nm<extra></extra>`
        };

        const shapes = [];
        const annotations = [];

        // Shared VNIR Region (430 - 860 nm)
        shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: 430,
            x1: 860,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(16, 185, 129, 0.12)',
            line: { width: 1.5, color: '#10b981' }
        });
        annotations.push({
            x: 645,
            y: 0.95,
            xref: 'x',
            yref: 'paper',
            text: 'Shared VNIR (430–860 nm)',
            showarrow: false,
            font: { size: 9, color: '#047857', family: 'sans-serif', weight: 'bold' },
            bgcolor: 'rgba(255,255,255,0.85)',
            bordercolor: '#10b981',
            borderwidth: 1
        });

        if (isGenuine) {
            // Missing SWIR Region in ROSIS (860 - 2500 nm)
            shapes.push({
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: 860,
                x1: 2500,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(239, 68, 68, 0.10)',
                line: { width: 1.5, color: '#ef4444', dash: 'dot' }
            });
            annotations.push({
                x: 1680,
                y: 0.85,
                xref: 'x',
                yref: 'paper',
                text: 'Missing SWIR Gap in ROSIS (860–2500 nm)',
                showarrow: false,
                font: { size: 9, color: '#b91c1c', family: 'sans-serif', weight: 'bold' },
                bgcolor: 'rgba(255,255,255,0.85)',
                bordercolor: '#ef4444',
                borderwidth: 1
            });

            // Key physical absorption band vertical lines
            const keyBands = [
                { nm: 705, label: "Red Edge (705nm)", color: "#059669" },
                { nm: 1400, label: "H₂O Notch (1400nm)", color: "#dc2626" },
                { nm: 1940, label: "Atm. H₂O (1940nm)", color: "#dc2626" },
                { nm: 2200, label: "Cellulose (2200nm)", color: "#dc2626" }
            ];
            keyBands.forEach(kb => {
                shapes.push({
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: kb.nm,
                    x1: kb.nm,
                    y0: 0,
                    y1: 0.70,
                    line: { color: kb.color, width: 1, dash: 'dot' }
                });
                annotations.push({
                    x: kb.nm,
                    y: 0.15,
                    xref: 'x',
                    yref: 'paper',
                    text: kb.label,
                    showarrow: false,
                    textangle: -90,
                    font: { size: 8.5, color: kb.color }
                });
            });
        }

        const layout = {
            title: {
                text: isGenuine
                    ? `<b>Optical Alignment: AVIRIS vs ROSIS</b>`
                    : `<b>Optical Alignment: AVIRIS (${resamplerData.source_dataset} vs ${resamplerData.target_dataset})</b>`,
                font: { size: 12, color: "#0f172a" },
                x: 0.05,
                y: 0.98
            },
            margin: { l: 55, r: 20, b: 40, t: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
                title: 'Wavelength (nm)',
                gridcolor: '#f1f5f9',
                tickfont: { size: 10, color: '#64748b' },
                range: [380, 2520]
            },
            yaxis: {
                title: 'Optical Sensitivity',
                gridcolor: '#f1f5f9',
                tickfont: { size: 10, color: '#64748b' },
                range: [-0.05, 1.15]
            },
            legend: {
                x: 0.02,
                y: 0.03,
                bgcolor: 'rgba(255, 255, 255, 0.92)',
                bordercolor: '#e2e8f0',
                borderwidth: 1,
                font: { size: 9.5 }
            },
            shapes: shapes,
            annotations: annotations
        };

        Plotly.react(chartEl, [traceAVIRIS, traceTargetSensor], layout, { responsive: true, displaylogo: false });
    }
}

function updateCrossSensorExplanations(pInfo, mSum, resamplerData) {
    const curvesEl = document.getElementById("csCurvesExplanation");
    const resamplerEl = document.getElementById("csResamplerExplanation");
    if (!pInfo && !mSum) return;

    const src = pInfo.source_dataset || mSum.source_dataset || "Source";
    const tgt = pInfo.target_dataset || mSum.target_dataset || "Target";
    const srcSensor = pInfo.source_sensor || mSum.source_sensor || "AVIRIS";
    const tgtSensor = pInfo.target_sensor || mSum.target_sensor || "AVIRIS";
    const isGenuine = (pInfo.is_genuine_cross_sensor === true) || (pInfo.transfer_type && pInfo.transfer_type.toLowerCase().includes("genuine"));
    const delta = mSum.transfer_delta_oa;
    const sign = delta > 0 ? "+" : "";

    if (curvesEl) {
        if (isGenuine) {
            curvesEl.innerHTML = `
                <strong>Empirical Finding (${src} &rarr; ${tgt}):</strong>
                Transferring from <em>${srcSensor}</em> (${src}) to <em>${tgtSensor}</em> (${tgt}) yields a final adaptation OA of <strong>${(mSum.transferred_oa || 0).toFixed(2)}%</strong> vs Target-only Pure Margin OA of <strong>${(mSum.target_oa || 0).toFixed(2)}%</strong> (&Delta;OA = <strong style="color:${delta > 0 ? 'var(--success)' : 'var(--danger)'};">${sign}${(delta || 0).toFixed(2)}%</strong>).
                Because the two sensors exhibit disparate ground sample distances (${pInfo.source_resolution_m}m vs ${pInfo.target_resolution_m}m) and non-overlapping spectral bands, transferred representations do not outperform a freshly initialized Pure Margin model on the target domain.
            `;
        } else {
            curvesEl.innerHTML = `
                <strong>Empirical Finding (${src} &rarr; ${tgt}):</strong>
                Transferring between same-sensor scenes (${srcSensor} &rarr; ${tgtSensor}) yields a final adaptation OA of <strong>${(mSum.transferred_oa || 0).toFixed(2)}%</strong> vs Target-only Pure Margin OA of <strong>${(mSum.target_oa || 0).toFixed(2)}%</strong> (&Delta;OA = <strong style="color:var(--success);">${sign}${(delta || 0).toFixed(2)}%</strong>).
                Source-pretrained latent feature representations transfer effectively across agricultural scenes, accelerating target classification accuracy in low-budget active learning rounds.
            `;
        }
    }

    if (resamplerEl) {
        if (isGenuine) {
            resamplerEl.innerHTML = `
                <strong>Optical Physics &amp; Sensor Calibration Disclosure:</strong>
                Plotted curves represent <em>calibrated sensor flight optical specifications</em> (no synthetic single-pixel predictions).
                AVIRIS captures <strong>400–2500 nm</strong> across 224 contiguous VNIR-SWIR channels, whereas ROSIS is physically confined to <strong>430–860 nm</strong> across 115 VNIR channels.
                While mathematical 1D spline interpolation can resample central wavelengths inside the shared <strong>430–860 nm</strong> VNIR window, ROSIS possesses <strong>zero physical detectors</strong> in the <strong>860–2500 nm</strong> SWIR region (where foliar water absorption at 1400/1940 nm and cellulose/lignin at 2200 nm reside).
                <strong>Interpolation cannot recreate physically unmeasured SWIR channels.</strong> This optical disparity, combined with the 15&times; spatial resolution mismatch (${pInfo.source_resolution_m || 20}m vs ${pInfo.target_resolution_m || 1.3}m GSD), accounts for the empirical negative transfer observed between AVIRIS and ROSIS.
            `;
        } else {
            resamplerEl.innerHTML = `
                <strong>Optical Physics &amp; Sensor Calibration Disclosure:</strong>
                Both source and target share identical AVIRIS sensor optics spanning the <strong>400–2500 nm</strong> VNIR-SWIR spectrum (100% optical compatibility).
                The 1D spline resampler seamlessly maps between the ${pInfo.source_bands || 200}-band and ${pInfo.target_bands || 204}-band sensor configurations with minimal residual gap (|&Delta;|&lt;0.01), providing complete optical alignment across the entire spectral range.
            `;
        }
    }
}

async function runLiveAdaptationTrial() {
    const btn = document.getElementById("btnRunAdaptationDemo");
    const banner = document.getElementById("liveTrialResultBanner");
    const title = document.getElementById("liveTrialTitle");
    const desc = document.getElementById("liveTrialDesc");
    const badge = document.getElementById("liveTrialBadge");
    const pairSel = document.getElementById("transferPairSelect");
    const pairKey = pairSel ? pairSel.value : "Indian Pines -> Salinas";

    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid #fff; border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite; vertical-align:middle; margin-right:6px;"></span> Running Trial...`;
    }

    try {
        const resp = await fetch("/api/run_adaptation_trial", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pair_key: pairKey,
                budget: 100,
                strategy: "margin",
                seed: 42
            })
        });
        const data = await resp.json();

        if (banner) banner.style.display = "block";
        if (title) title.innerText = `Live Single-Trial Test Result: ${data.pair_key} (Budget N=${data.budget}, Seed ${data.seed})`;

        if (desc) {
            const deltaSign = data.delta_oa > 0 ? "+" : "";
            const deltaColor = data.delta_oa > 0 ? "var(--success)" : (data.delta_oa < 0 ? "var(--danger)" : "var(--text-secondary)");
            desc.innerHTML = `
                Evaluated in <strong>${data.train_time_s.toFixed(3)}s</strong>.
                Transferred Pure Margin OA: <strong>${data.transferred_oa.toFixed(2)}%</strong> vs Target-only Pure Margin OA: <strong>${data.target_only_oa.toFixed(2)}%</strong>
                (&Delta;OA = <strong style="color:${deltaColor};">${deltaSign}${data.delta_oa.toFixed(2)}%</strong>).
                <em>Outcome: ${data.outcome}</em>.
            `;
        }

        if (badge) {
            if (data.outcome.toLowerCase().includes("positive")) {
                badge.className = "badge badge-success";
                badge.innerText = `Positive Transfer (${data.delta_oa > 0 ? '+' : ''}${data.delta_oa.toFixed(2)}%)`;
            } else if (data.outcome.toLowerCase().includes("negative")) {
                badge.className = "badge badge-danger";
                badge.innerText = `Negative Transfer (${data.delta_oa.toFixed(2)}%)`;
            } else {
                badge.className = "badge badge-secondary";
                badge.innerText = `Neutral (${data.delta_oa > 0 ? '+' : ''}${data.delta_oa.toFixed(2)}%)`;
            }
        }
    } catch (err) {
        console.error("Live adaptation trial failed:", err);
        if (banner) banner.style.display = "block";
        if (desc) desc.innerText = `Error executing trial: ${err.message}`;
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `⚡ Run Live Adaptation Trial`;
        }
    }
}
window.runLiveAdaptationTrial = runLiveAdaptationTrial;

/* =========================================================================
   DATASET EXPLORER (SECTION C)
   ========================================================================= */
let _datasetDetailsCache = {};
async function switchDatasetExplorerView(dsName) {
    window._currentExplorerDs = dsName;
    window._datasetExplorerLoaded = true;

    const btns = document.querySelectorAll("#datasetExplorerTabs .segmented-btn");
    btns.forEach(b => {
        if (b.innerText.includes(dsName)) b.classList.add("active");
        else b.classList.remove("active");
    });

    const container = document.getElementById("datasetExplorerContent");
    if (!container) return;

    try {
        if (!_datasetDetailsCache[dsName]) {
            const res = await fetch(`/api/dataset_details?dataset_name=${encodeURIComponent(dsName)}`);
            _datasetDetailsCache[dsName] = await res.json();
        }
        const data = _datasetDetailsCache[dsName];

        const isAviris = data.sensor === "AVIRIS";
        const hasSwir = isAviris;
        const badgeColor = isAviris ? "badge-primary" : "badge-warning";
        const vnirFlex = isAviris ? "1" : "2";

        container.innerHTML = `
            <div class="card" style="background:#ffffff; border:1px solid var(--border-color); border-radius:8px; padding:18px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:14px;">
                    <div>
                        <h3 style="font-size:17px; font-weight:800; color:var(--text-primary); margin:0 0 4px 0;">${data.name}</h3>
                        <span class="badge ${badgeColor}">${data.sensor} (${data.sensor_family})</span>
                        <span style="font-size:12px; color:var(--text-muted); margin-left:8px;">${data.environment} Landscape</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:12px; color:var(--text-secondary);">Receptive Field Patch:</span>
                        <strong style="font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--primary); margin-left:4px;">15 &times; 15 &times; ${data.spectral_bands}</strong>
                    </div>
                </div>

                <div class="grid-4" style="gap:10px;">
                    <div class="metric-card">
                        <span class="metric-title">Spectral Coverage</span>
                        <div class="metric-value" style="font-size:15px;">${data.spectral_coverage}</div>
                        <div class="metric-sub">${data.spectral_bands} Calibrated Bands (${data.spectral_resolution_nm})</div>
                    </div>
                    <div class="metric-card">
                        <span class="metric-title">Spatial Resolution</span>
                        <div class="metric-value" style="font-size:15px;">${data.spatial_resolution_m}</div>
                        <div class="metric-sub">Ground Sampling Distance (GSD)</div>
                    </div>
                    <div class="metric-card">
                        <span class="metric-title">Spatial Grid & Pixels</span>
                        <div class="metric-value" style="font-size:15px;">${data.dimensions[0]} &times; ${data.dimensions[1]}</div>
                        <div class="metric-sub">${data.labeled_pixels.toLocaleString()} labeled / ${data.total_pixels.toLocaleString()} total</div>
                    </div>
                    <div class="metric-card">
                        <span class="metric-title">Land-Cover Taxonomy</span>
                        <div class="metric-value highlight" style="font-size:15px;">${data.num_classes} Classes</div>
                        <div class="metric-sub">${data.unlabeled_background_pixels.toLocaleString()} background pixels</div>
                    </div>
                </div>

                <div style="margin-top:14px; padding:12px 14px; background:var(--bg-subtle); border-radius:6px; border:1px solid var(--border-color);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:12px; font-weight:700; color:var(--text-primary);">Physical Optical Spectrum Profile</span>
                        <span style="font-size:11px; font-family:'JetBrains Mono',monospace; color:var(--text-muted);">400 nm &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash; 2500 nm</span>
                    </div>
                    <div class="spectrum-range-bar-container">
                        <div style="display:flex; height:22px; border-radius:4px; overflow:hidden; border:1px solid rgba(0,0,0,0.1);">
                            <div style="flex:${vnirFlex}; background:linear-gradient(90deg, #3b82f6, #10b981); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-weight:700;" title="VNIR (400-1000 nm)">VNIR (400–1000 nm)</div>
                            ${hasSwir ? `<div style="flex:1.5; background:linear-gradient(90deg, #f59e0b, #ef4444); display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:10px; font-weight:700;" title="SWIR (1000-2500 nm)">SWIR (1000–2500 nm)</div>` : `<div style="flex:1.5; background:#cbd5e1; display:flex; align-items:center; justify-content:center; color:#64748b; font-size:10px; font-weight:700;" title="No SWIR Channels">NO SWIR COVERAGE (BLIND &gt;860 nm)</div>`}
                        </div>
                    </div>
                    <p style="font-size:11.5px; color:var(--text-secondary); margin:8px 0 0 0; line-height:1.4;">${data.optical_analysis}</p>
                </div>

                <div style="margin-top:14px;">
                    <h4 style="font-size:12.5px; font-weight:700; color:var(--text-primary); margin-bottom:8px;">Ground-Truth Land-Cover Categories (${data.num_classes} Classes)</h4>
                    <div class="class-chips-grid">
                        ${data.classes.map(c => `
                            <div class="class-chip">
                                <span class="class-chip-color" style="background:${c.color};"></span>
                                <span style="font-size:10.5px; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">#${c.id}</span>
                                <span style="font-weight:600; color:var(--text-primary);">${c.name}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    } catch (err) {
        console.error("Error loading dataset details:", err);
    }
}

/* =========================================================================
   ACTIVE LEARNING BENCHMARKS (SECTION D)
   ========================================================================= */
let _benchmarkDataCache = {};
let _statResultsCache = null;

async function loadBenchmarkData() {
    window._benchmarkDataLoaded = true;
    const protoSel = document.getElementById("benchmarkProtocolSelect");
    const protocol = protoSel ? protoSel.value : "random_pixel";

    try {
        if (!_benchmarkDataCache[protocol]) {
            const res = await fetch(`/api/method_comparison_data?protocol=${encodeURIComponent(protocol)}`);
            _benchmarkDataCache[protocol] = await res.json();
        }
        if (!_statResultsCache) {
            const resStat = await fetch("/api/statistical_results");
            _statResultsCache = await resStat.json();
        }
        if (!_labelEffCache[protocol]) {
            const resEff = await fetch(`/api/label_efficiency_data?protocol=${encodeURIComponent(protocol)}`);
            _labelEffCache[protocol] = await resEff.json();
        }
        if (_labelEffCache[protocol] && _labelEffCache[protocol].thresholds_table) {
            renderThresholdsTable(_labelEffCache[protocol].thresholds_table);
        }
        calculateLabelCost();
        renderBenchmarkTab();
    } catch (err) {
        console.error("Error loading benchmark data:", err);
    }
}

function renderBenchmarkTab() {
    const protoSel = document.getElementById("benchmarkProtocolSelect");
    const dsSel = document.getElementById("benchmarkDatasetSelect");
    const metricSel = document.getElementById("benchmarkMetricSelect");
    const chkShowAll = document.getElementById("chkShowAllMethods");

    const protocol = protoSel ? protoSel.value : "random_pixel";
    const dsName = dsSel ? dsSel.value : "Indian Pines";
    const metric = metricSel ? metricSel.value : "oa";
    const showAll = chkShowAll ? chkShowAll.checked : false;

    const dataObj = _benchmarkDataCache[protocol];
    if (!dataObj || !dataObj.datasets || !dataObj.datasets[dsName]) return;

    const fullList = dataObj.datasets[dsName];
    let chartList;
    if (showAll) {
        chartList = fullList.filter(st => {
            const name = st.strategy;
            return name.includes("Pure Margin") ||
                   name.includes("Random Sampling") ||
                   name.includes("Least Confidence") ||
                   name.includes("Margin + FD") ||
                   name.includes("ConformAL") ||
                   name.includes("ACBM");
        });
    } else {
        chartList = fullList.filter(st => {
            const name = st.strategy;
            return name.includes("Pure Margin") || name.includes("Random Sampling");
        });
    }

    const chartEl = document.getElementById("benchmarkChart");
    if (chartEl) showGraphGeneratingAnimation(chartEl.parentElement || chartEl, "Generating Benchmark Trajectories...", "Comparing Pure Margin vs Historical Baselines");
    const is3D = (chartModes.benchmark === "3d");

    const colorMap = {
        "Pure Margin Sampling": "#2563eb",
        "Pure Margin": "#2563eb",
        "Random Sampling": "#475569",
        "Least Confidence": "#f59e0b",
        "Margin + FD (Euclidean)": "#10b981",
        "Margin + Cosine FD": "#059669",
        "Margin + Cosine Diversity": "#047857",
        "Integrated ConformAL-HSI": "#8b5cf6",
        "ConformAL-HSI (Integrated)": "#8b5cf6",
        "ACBM-Diversity (Historical Benchmark)": "#ec4899",
        "LC + FD (Euclidean)": "#f97316",
        "Farthest-First Diversity": "#d97706"
    };

    if (chartEl && window.Plotly) {
        if (is3D) {
            const traces = chartList.map((st, idx) => {
                const xVals = st.points.map(p => p.budget);
                const yVals = Array(st.points.length).fill(idx);
                const zVals = st.points.map(p => p[metric] !== undefined ? p[metric] : p.oa);
                const col = colorMap[st.strategy] || "#3b82f6";
                const isPrimary = st.is_primary || st.strategy.includes("Pure Margin");
                return {
                    type: "scatter3d",
                    mode: "lines+markers",
                    name: st.strategy,
                    x: xVals,
                    y: yVals,
                    z: zVals,
                    line: { color: col, width: isPrimary ? 7 : 4, dash: st.is_historical ? "dash" : "solid" },
                    marker: { size: isPrimary ? 6 : 4, color: col }
                };
            });
            const layout = {
                margin: { l: 0, r: 0, b: 0, t: 0 },
                paper_bgcolor: "rgba(0,0,0,0)",
                scene: {
                    xaxis: { title: "Budget (N)", gridcolor: "#e2e8f0" },
                    yaxis: { title: "Strategy", tickvals: chartList.map((_, i) => i), ticktext: chartList.map(s => s.strategy.split(" ")[0]), gridcolor: "#e2e8f0" },
                    zaxis: { title: `${metric.toUpperCase()}`, gridcolor: "#e2e8f0" },
                    camera: { eye: { x: 1.6, y: -1.7, z: 1.2 } }
                },
                legend: { x: 0.02, y: 0.98, font: { size: 10 } }
            };
            Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
        } else {
            const traces = chartList.map(st => {
                const xVals = st.points.map(p => p.budget);
                const yVals = st.points.map(p => p[metric] !== undefined ? p[metric] : p.oa);
                const col = colorMap[st.strategy] || "#3b82f6";
                const isPrimary = st.is_primary || st.strategy.includes("Pure Margin");
                const isRandom = st.strategy.includes("Random");

                return {
                    type: "scatter",
                    mode: "lines+markers",
                    name: isPrimary ? `${st.strategy} (Primary Proposed)` : (isRandom ? `${st.strategy} (Passive Baseline)` : st.strategy),
                    x: xVals,
                    y: yVals,
                    line: {
                        color: col,
                        width: isPrimary ? 3.5 : (isRandom ? 2.2 : 1.8),
                        dash: isPrimary ? "solid" : (isRandom ? "dot" : (st.is_historical ? "dash" : "dashdot"))
                    },
                    marker: {
                        size: isPrimary ? 7 : (isRandom ? 5 : 4),
                        symbol: isPrimary ? "diamond" : (isRandom ? "circle" : "square")
                    }
                };
            });
            const layout = {
                margin: { l: 45, r: 20, b: 35, t: 20 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                xaxis: { title: "Labeled Budget (N)", gridcolor: "#f1f5f9", tickfont: { size: 10, color: "#64748b" } },
                yaxis: { title: `${metric.toUpperCase()} (%)`, gridcolor: "#f1f5f9", tickfont: { size: 10, color: "#64748b" } },
                legend: { x: 0.02, y: 0.98, font: { size: 10 } }
            };
            Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
        }
        setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 250);
    }

    const tbody = document.querySelector("#benchmarkMetricsTable tbody");
    if (tbody) {
        tbody.innerHTML = "";
        fullList.forEach(st => {
            const p500 = st.points.find(p => p.budget === 500);
            const val500 = p500 ? `${p500[metric] !== undefined ? p500[metric].toFixed(2) : p500.oa.toFixed(2)}%` : "--";
            const finalP = st.points[st.points.length - 1];
            const finalVal = finalP ? `${finalP[metric] !== undefined ? finalP[metric].toFixed(2) : finalP.oa.toFixed(2)}%` : "--";

            const isPrimary = st.is_primary || st.strategy.includes("Pure Margin");
            let roleBadge = `<span class="badge badge-secondary">Baseline</span>`;
            if (isPrimary) roleBadge = `<span class="badge-primary-method">Primary Proposed</span>`;
            else if (st.is_historical) roleBadge = `<span class="badge-historical">Historical Comparison</span>`;

            const tr = document.createElement("tr");
            if (isPrimary) tr.style.background = "rgba(37, 99, 235, 0.06)";
            tr.innerHTML = `
                <td><strong>${st.strategy}</strong></td>
                <td>${roleBadge}</td>
                <td><strong style="font-family:'JetBrains Mono',monospace;">${st.aucl.toFixed(2)}</strong></td>
                <td><span style="font-family:'JetBrains Mono',monospace;">${val500}</span></td>
                <td><strong style="font-family:'JetBrains Mono',monospace; color:${isPrimary ? 'var(--primary)' : 'inherit'};">${finalVal}</strong></td>
            `;
            tbody.appendChild(tr);
        });
    }

    const statTbody = document.querySelector("#statTestsTable tbody");
    if (statTbody && _statResultsCache && _statResultsCache.active_learning_tests) {
        statTbody.innerHTML = "";
        const tests = _statResultsCache.active_learning_tests.filter(t => t.dataset === dsName && t.protocol === protocol);
        tests.forEach(t => {
            const tr = document.createElement("tr");
            const sigBadge = t.significant ? `<span class="badge badge-success">Statistically Significant</span>` : `<span class="badge badge-warning">Not Significant</span>`;
            tr.innerHTML = `
                <td>${t.dataset}</td>
                <td>${t.protocol}</td>
                <td>${t.budget}</td>
                <td><strong>${t.mean_diff_pct > 0 ? '+' : ''}${t.mean_diff_pct.toFixed(2)}%</strong></td>
                <td>${t.t_stat.toFixed(3)}</td>
                <td>${t.p_uncorrected < 0.001 ? '<0.001' : t.p_uncorrected.toFixed(4)}</td>
                <td>${t.cohens_d.toFixed(2)}</td>
                <td>${sigBadge}</td>
            `;
            statTbody.appendChild(tr);
        });
    }
}

/* =========================================================================
   LABEL EFFICIENCY & WORKLOAD SAVINGS (SECTION F)
   ========================================================================= */
let _labelEffCache = {};

async function switchLabelEffProtocol(proto) {
    window._currentLabelEffProto = proto;
    window._labelEffLoaded = true;

    const btns = document.querySelectorAll("#labelEffProtocolToggle .segmented-btn");
    btns.forEach(b => {
        if ((proto === "random_pixel" && b.innerText.includes("Random")) ||
            (proto === "spatially_disjoint" && b.innerText.includes("Spatially"))) {
            b.classList.add("active");
        } else {
            b.classList.remove("active");
        }
    });

    try {
        if (!_labelEffCache[proto]) {
            const res = await fetch(`/api/label_efficiency_data?protocol=${encodeURIComponent(proto)}`);
            _labelEffCache[proto] = await res.json();
        }
        const data = _labelEffCache[proto];

        renderLabelEfficiencyChart(data.curves_data);
        renderThresholdsTable(data.thresholds_table);
        calculateLabelCost();

    } catch (err) {
        console.error("Error switching label efficiency protocol:", err);
    }
}

function renderLabelEfficiencyChart(curvesData) {
    const chartEl = document.getElementById("labelEfficiencyChart");
    if (!chartEl || !window.Plotly || !curvesData) return;
    showGraphGeneratingAnimation(chartEl.parentElement || chartEl, "Computing Operational Label Savings...", "Target Accuracy vs Certified Annotation Count");

    const is3D = (chartModes.labelEff === "3d");
    const datasets = Object.keys(curvesData);

    if (is3D) {
        const traces = [];
        datasets.forEach((ds, dsIdx) => {
            const d = curvesData[ds];
            const counts = d.counts;
            traces.push({
                type: "scatter3d",
                mode: "lines+markers",
                name: `${ds} (Random)`,
                x: counts,
                y: Array(counts.length).fill(dsIdx * 2),
                z: d["Random"],
                line: { color: "#64748b", width: 4, dash: "dash" },
                marker: { size: 4, color: "#64748b" }
            });
            traces.push({
                type: "scatter3d",
                mode: "lines+markers",
                name: `${ds} (Pure Margin)`,
                x: counts,
                y: Array(counts.length).fill(dsIdx * 2 + 1),
                z: d["Pure Margin"] || d["Margin"],
                line: { color: "#2563eb", width: 6 },
                marker: { size: 5, color: "#2563eb" }
            });
        });
        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "rgba(0,0,0,0)",
            scene: {
                xaxis: { title: "Labeled Samples", gridcolor: "#e2e8f0" },
                yaxis: { title: "Dataset / Strategy", gridcolor: "#e2e8f0" },
                zaxis: { title: "Overall Accuracy (%)", gridcolor: "#e2e8f0" },
                camera: { eye: { x: 1.6, y: -1.7, z: 1.2 } }
            },
            legend: { x: 0.02, y: 0.98, font: { size: 10 } }
        };
        Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
    } else {
        const traces = [];
        const colors = { "Indian Pines": "#2563eb", "Salinas": "#10b981", "Pavia University": "#f59e0b" };
        datasets.forEach(ds => {
            const d = curvesData[ds];
            const col = colors[ds] || "#3b82f6";
            traces.push({
                type: "scatter",
                mode: "lines+markers",
                name: `${ds} (Pure Margin)`,
                x: d.counts,
                y: d["Pure Margin"] || d["Margin"],
                line: { color: col, width: 3 },
                marker: { size: 6, symbol: "diamond" }
            });
            traces.push({
                type: "scatter",
                mode: "lines",
                name: `${ds} (Random)`,
                x: d.counts,
                y: d["Random"],
                line: { color: col, width: 1.5, dash: "dot" },
                marker: { size: 4 }
            });
        });
        const layout = {
            margin: { l: 45, r: 20, b: 35, t: 20 },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            xaxis: { title: "Labeled Samples Count", gridcolor: "#f1f5f9", tickfont: { size: 10, color: "#64748b" } },
            yaxis: { title: "Overall Accuracy (OA %)", gridcolor: "#f1f5f9", tickfont: { size: 10, color: "#64748b" } },
            legend: { x: 0.02, y: 0.98, font: { size: 10 } }
        };
        Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
        setTimeout(() => hideGraphGeneratingAnimation(chartEl.parentElement || chartEl), 250);
    }
}

function renderThresholdsTable(rows) {
    const tbody = document.querySelector("#thresholdsTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    if (!rows || !Array.isArray(rows) || rows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted);">No threshold data available for this protocol.</td></tr>`;
        return;
    }

    rows.forEach(r => {
        const tr = document.createElement("tr");

        const ds = r.Dataset || r.dataset || "N/A";
        const targetOA = r.Target_OA || (r.target_accuracy !== undefined ? `${Number(r.target_accuracy).toFixed(2)}%` : "N/A");

        const randLabels = (r.Random_Labels !== undefined && r.Random_Labels !== null)
            ? Number(r.Random_Labels)
            : (r.random_budget !== undefined ? Number(r.random_budget) : null);

        const marginLabels = (r.Margin_Labels !== undefined && r.Margin_Labels !== null)
            ? Number(r.Margin_Labels)
            : ((r.Margin_Labels_Needed !== undefined && r.Margin_Labels_Needed !== null)
                ? Number(r.Margin_Labels_Needed)
                : (r.discrete_budget_needed !== undefined ? Number(r.discrete_budget_needed) : null));

        let savedLabels = null;
        if (r.Saved_Labels !== undefined && r.Saved_Labels !== null) {
            savedLabels = Number(r.Saved_Labels);
        } else if (r.discrete_labels_saved !== undefined && r.discrete_labels_saved !== null) {
            savedLabels = Number(r.discrete_labels_saved);
        } else if (randLabels !== null && marginLabels !== null) {
            savedLabels = randLabels - marginLabels;
        }

        let pctSaved = null;
        if (r.Percentage_Saved !== undefined && r.Percentage_Saved !== null && !isNaN(r.Percentage_Saved)) {
            pctSaved = Number(r.Percentage_Saved);
        } else if (r.discrete_percent_saved !== undefined && r.discrete_percent_saved !== null && !isNaN(r.discrete_percent_saved)) {
            pctSaved = Number(r.discrete_percent_saved);
        } else if (r.Percent_Saved && typeof r.Percent_Saved === "string") {
            const parsed = parseFloat(r.Percent_Saved.replace("%", ""));
            if (!isNaN(parsed)) pctSaved = parsed;
        } else if (savedLabels !== null && randLabels !== null && randLabels > 0) {
            pctSaved = (savedLabels / randLabels) * 100;
        }

        const randStr = (randLabels !== null && !isNaN(randLabels)) ? `${randLabels}` : "N/A";
        const marginStr = (marginLabels !== null && !isNaN(marginLabels)) ? `<strong>${marginLabels}</strong>` : "N/A";

        let savedStr = "N/A";
        let badgeHtml = `<span class="badge badge-secondary">0.0%</span>`;

        if (savedLabels !== null && !isNaN(savedLabels)) {
            if (savedLabels > 0) {
                savedStr = `<strong style="color:var(--success);">+${savedLabels}</strong>`;
            } else if (savedLabels < 0) {
                savedStr = `<strong style="color:var(--danger);">${savedLabels}</strong>`;
            } else {
                savedStr = `<span style="color:var(--text-muted);">0</span>`;
            }
        }

        if (pctSaved !== null && !isNaN(pctSaved)) {
            const formattedPct = `${pctSaved > 0 ? '+' : ''}${pctSaved.toFixed(1)}%`;
            if (pctSaved > 0) {
                badgeHtml = `<span class="badge badge-success">${formattedPct}</span>`;
            } else if (pctSaved < 0) {
                badgeHtml = `<span class="badge badge-danger">${formattedPct}</span>`;
            } else {
                badgeHtml = `<span class="badge badge-secondary">0.0%</span>`;
            }
        }

        tr.innerHTML = `
            <td><strong>${ds}</strong></td>
            <td><span class="badge badge-primary">${targetOA}</span></td>
            <td>${randStr}</td>
            <td>${marginStr}</td>
            <td>${savedStr}</td>
            <td>${badgeHtml}</td>
        `;
        tbody.appendChild(tr);
    });
}

/* =========================================================================
   CLASSIFICATION RESULTS & CONFUSION MATRIX (SECTION H)
   ========================================================================= */
let _cmCache = {};

async function loadClassificationResults() {
    window._classificationLoaded = true;
    const dsSel = document.getElementById("cmDatasetSelect");
    const protoSel = document.getElementById("cmProtocolSelect");
    const stratSel = document.getElementById("cmStrategySelect");

    const dataset = dsSel ? dsSel.value : "Indian Pines";
    const protocol = protoSel ? protoSel.value : "random_pixel";
    const strategy = stratSel ? stratSel.value : "margin";

    const cacheKey = `${dataset}_${protocol}_${strategy}`;
    try {
        if (!_cmCache[cacheKey]) {
            const res = await fetch(`/api/classification_results?dataset=${encodeURIComponent(dataset)}&protocol=${encodeURIComponent(protocol)}&strategy=${encodeURIComponent(strategy)}`);
            _cmCache[cacheKey] = await res.json();
        }
        const data = _cmCache[cacheKey];

        const cmEl = document.getElementById("confusionMatrixChart");
        if (cmEl && window.Plotly && data.confusion_matrix && data.confusion_matrix.length > 0) {
            showGraphGeneratingAnimation(cmEl.parentElement || cmEl, "Synthesizing Per-Class Confusion Matrix...", "Pavia University 9x9 Empirical Prediction Matrix");
            const trace = {
                type: "heatmap",
                z: data.confusion_matrix,
                x: data.class_names,
                y: data.class_names,
                colorscale: "Blues",
                reversescale: false,
                hovertemplate: "Ground Truth: %{y}<br>Predicted: %{x}<br>Rate: %{z:.2%}<extra></extra>"
            };
            const layout = {
                margin: { l: 120, r: 20, b: 100, t: 20 },
                xaxis: { title: "Predicted Class", tickangle: -45, tickfont: { size: 9 } },
                yaxis: { title: "Ground Truth Class", tickfont: { size: 9 }, autorange: "reversed" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)"
            };
            Plotly.react(cmEl, [trace], layout, { responsive: true, displaylogo: false });
            setTimeout(() => hideGraphGeneratingAnimation(cmEl.parentElement || cmEl), 250);
        }

        const tbody = document.querySelector("#perClassMetricsTable tbody");
        if (tbody && data.per_class_metrics) {
            tbody.innerHTML = "";
            data.per_class_metrics.forEach(c => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td>
                        <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:${c.color}; margin-right:6px; vertical-align:middle;"></span>
                        <strong>${c.class_name}</strong>
                    </td>
                    <td>${c.precision.toFixed(1)}%</td>
                    <td>${c.recall.toFixed(1)}%</td>
                    <td><strong>${c.f1.toFixed(1)}%</strong></td>
                    <td>${c.support}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Error loading classification results:", err);
    }
}

/* =========================================================================
   RESEARCH EXPORTS
   ========================================================================= */
async function exportExperimentSession() {
    try {
        const res = await fetch("/api/export_experiment", { method: "POST" });
        const data = await res.json();

        alert(`Session exported successfully!\n\nJSON: ${data.json_path}\nCSV: ${data.csv_path}`);
    } catch (err) {
        console.error("Error exporting experiment:", err);
    }
}

/* =========================================================================
   SAMPLE MODAL 2D SPECTRUM CHART
   ========================================================================= */
function renderSampleSpectrumChart(data) {
    const spectrumEl = document.getElementById("sampleSpectrumChart");
    if (!spectrumEl) return;
    lastSampleData = data;

    const hasWl = (data.wavelengths && Array.isArray(data.wavelengths) && data.wavelengths.length > 0);
    const xLabels = hasWl ? data.wavelengths.map(w => Math.round(w)) : Array.from({length: data.spectrum.length}, (_, i) => i + 1);
    const xTitle = hasWl ? 'λ (nm)' : 'Spectral Band Index';

    if (window.Plotly) {
        const trace = {
            type: 'scatter',
            mode: 'lines',
            name: 'Reflectance Signature',
            x: xLabels,
            y: data.spectrum,
            line: { color: '#2563eb', width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(37, 99, 235, 0.08)',
            hovertemplate: hasWl
                ? '<b>Spectral Signature</b><br>Wavelength: %{x} nm<br>Response: %{y:.4f}<extra></extra>'
                : '<b>Spectral Signature</b><br>Band: %{x}<br>Response: %{y:.4f}<extra></extra>'
        };

        const layout = {
            margin: { l: 40, r: 15, b: 30, t: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
                title: { text: xTitle, font: { size: 10, color: '#64748b' } },
                gridcolor: '#f1f5f9',
                tickfont: { size: 9, color: '#64748b' }
            },
            yaxis: {
                title: { text: 'Reflectance', font: { size: 10, color: '#64748b' } },
                gridcolor: '#f1f5f9',
                tickfont: { size: 9, color: '#64748b' }
            },
            showlegend: false
        };

        Plotly.react(spectrumEl, [trace], layout, { responsive: true, displaylogo: false });
    }
}

/* =========================================================================
   INTERACTIVE 3D HYPERSPECTRAL SCENE & QUERIED SAMPLE SYSTEM
   ========================================================================= */
// currentSceneViewMode hoisted to top globals
let isScene3DAutoRotating = false;
let scene3DRotationRafId = null;
let scene3DCameraAngle = -Math.PI / 2;
let show3DSamples = true;

function isWebGLAvailable() {
    try {
        const canvas = document.createElement('canvas');
        return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) {
        return false;
    }
}

function setSceneViewMode(mode) {
    currentSceneViewMode = mode;
    const btn3D = document.getElementById("btnSceneMode3D");
    const btn2D = document.getElementById("btnSceneMode2D");
    const container3D = document.getElementById("scene3DContainer");
    const container2D = document.getElementById("scene2DContainer");
    const mapControls = document.getElementById("sceneMapControls");

    if (mode === '3d') {
        if (btn3D) btn3D.classList.add("active");
        if (btn2D) btn2D.classList.remove("active");
        if (container3D) container3D.style.display = "block";
        if (container2D) container2D.style.display = "none";
        if (mapControls) mapControls.style.display = "none";
        renderScene3D();
    } else {
        if (btn2D) btn2D.classList.add("active");
        if (btn3D) btn3D.classList.remove("active");
        if (container2D) container2D.style.display = "block";
        if (container3D) container3D.style.display = "none";
        if (mapControls && hasPredicted) mapControls.style.display = "flex";
        stopScene3DAutoRotate();
        renderSpatialCanvas();
    }
}

function update3DLayers() {
    const togglePred = document.getElementById("toggle3dPredictions");
    const toggleHuman = document.getElementById("toggle3dHuman");
    const togglePseudo = document.getElementById("toggle3dPseudo");
    const toggleQueried = document.getElementById("toggle3dQueried");

    show3DPredictions = togglePred ? togglePred.checked : true;
    show3DHuman = toggleHuman ? toggleHuman.checked : true;
    show3DPseudo = togglePseudo ? togglePseudo.checked : true;
    show3DQueried = toggleQueried ? toggleQueried.checked : true;

    renderScene3D();
}

function update3DStatsPanel() {
    try {
        const statHuman = document.getElementById("stat3dHumanCount");
        const statPred = document.getElementById("stat3dPredCount");
        const statPseudo = document.getElementById("stat3dPseudoCount");
        const statOA = document.getElementById("stat3dTestOA");
        const badgeEl = document.getElementById("scene3dSamplesBadge");

        let humanCount = 160;
        let predCount = 0;
        let pseudoCount = 0;
        let roundNum = 0;
        let oaVal = "--";

        if (sceneData) {
            humanCount = sceneData.num_labeled || 160;
        }

        if (lastPredictionsData && lastPredictionsData.statistics) {
            const s = lastPredictionsData.statistics;
            humanCount = s.human_labeled_samples || humanCount;
            pseudoCount = s.accepted_pseudo_labels || 0;
            roundNum = s.round || 0;
            if (hasRetrained || roundNum > 0) {
                predCount = s.predicted_pixels || 0;
            } else {
                predCount = 0;
            }
            if (s.current_test_accuracy > 0) {
                oaVal = `${s.current_test_accuracy.toFixed(2)}%`;
            }
        }

        if (oaVal === "--") {
            const metricOaEl = document.getElementById("metricOA");
            if (metricOaEl && metricOaEl.innerText && metricOaEl.innerText !== "--") {
                oaVal = metricOaEl.innerText;
            }
        }

        if (statHuman) statHuman.innerText = humanCount;
        if (statPred) statPred.innerText = predCount > 0 ? predCount.toLocaleString() : "0";
        if (statPseudo) statPseudo.innerText = pseudoCount;
        if (statOA) statOA.innerText = oaVal;

        if (badgeEl) {
            const hasSamples = querySelectionData && querySelectionData.selected_samples && querySelectionData.selected_samples.length > 0;
            if (hasSamples) {
                badgeEl.innerHTML = `Round ${roundNum}: <strong>${querySelectionData.selected_samples.length} Queried Candidates</strong> (#1–#20)`;
            } else if (lastPredictionsData && lastPredictionsData.prediction_map) {
                badgeEl.innerHTML = `Post-Annotation 3D Classification Map &bull; Round ${roundNum} &bull; ${predCount.toLocaleString()} Pixels Classified`;
            } else {
                badgeEl.innerHTML = `3D Hyperspectral Dataset View &bull; ${humanCount} Labeled Seeds`;
            }
        }
    } catch (err) {
        console.error("Error updating 3D stats panel:", err);
    }
}

function showGeneric3DInfo(htmlContent) {
    const infoEl = document.getElementById("scene3dSelectedInfo");
    if (!infoEl) return;
    infoEl.innerHTML = `<div style="font-size:11.5px; line-height:1.5;">${htmlContent}</div>`;
}

function renderScene3D() {
    if (!sceneData || !window.Plotly) return;
    const sceneEl = document.getElementById("hsiSpatialScene3D");
    if (!sceneEl) return;
    showGraphGeneratingAnimation(sceneEl.parentElement || sceneEl, "Rendering 3D Hyperspectral Data Cube...", "Volumetric Reflectance & Active Query Markers");

    // Immediately update stats panel on every render
    update3DStatsPanel();

    const [height, width] = sceneData.dimensions;

    // Subsample high-resolution scenes for 60fps WebGL responsiveness
    let step = 1;
    if (width > 200 || height > 200) {
        step = Math.max(1, Math.floor(Math.max(width, height) / 220));
    }

    const xGrid = [];
    for (let c = 0; c < width; c += step) {
        xGrid.push(c);
    }
    if (xGrid[xGrid.length - 1] !== width - 1) xGrid.push(width - 1);

    const yGrid = [];
    for (let r = 0; r < height; r += step) {
        yGrid.push(r);
    }
    if (yGrid[yGrid.length - 1] !== height - 1) yGrid.push(height - 1);

    const M = yGrid.length;
    const N = xGrid.length;
    const zGrid = new Array(M);
    const colorGrid = new Array(M);

    const hasPredictions = lastPredictionsData && lastPredictionsData.prediction_map;
    const roundNum = lastPredictionsData?.statistics?.round || 0;
    const isPredictMode = (hasRetrained || roundNum > 0) && show3DPredictions && hasPredictions;

    let colorscale = sceneData.scene_colorscale;
    let cmin = 0;
    let cmax = 255;

    if (isPredictMode) {
        // Build discrete step colorscale mapping each eligible pixel to its predicted land-cover class color
        const numClasses = sceneData.num_classes || 16;
        const totalBuckets = numClasses + 1; // Bucket 0: background, Buckets 1..C: classes
        const classColors = sceneData.class_colors || [];

        colorscale = [];
        const bgRgb = 'rgb(15, 23, 42)'; // subtle dark background for non-ground-truth pixels
        colorscale.push([0, bgRgb]);
        colorscale.push([1 / totalBuckets, bgRgb]);

        for (let c = 0; c < numClasses; c++) {
            const rgbColor = hexToRgb(classColors[c] || '#2563eb');
            const startVal = (c + 1) / totalBuckets;
            const endVal = (c + 2) / totalBuckets;
            colorscale.push([startVal, rgbColor]);
            colorscale.push([endVal, rgbColor]);
        }
        cmin = 0;
        cmax = totalBuckets;

        const predMap = lastPredictionsData.prediction_map;
        for (let i = 0; i < M; i++) {
            const r = yGrid[i];
            const zRow = new Array(N);
            const cRow = new Array(N);
            const pRow = predMap ? predMap[r] : null;
            for (let j = 0; j < N; j++) {
                const c = xGrid[j];
                zRow[j] = 0;
                const predVal = pRow ? pRow[c] : -1;
                if (predVal < 0) {
                    cRow[j] = 0.5; // background bucket
                } else {
                    cRow[j] = predVal + 1.5; // class bucket
                }
            }
            zGrid[i] = zRow;
            colorGrid[i] = cRow;
        }
    } else {
        // Initial authentic hyperspectral false-color composite
        const rawIndices = sceneData.scene_color_indices;
        for (let i = 0; i < M; i++) {
            const r = yGrid[i];
            const zRow = new Array(N);
            const cRow = new Array(N);
            const rawRow = rawIndices ? rawIndices[r] : null;
            for (let j = 0; j < N; j++) {
                const c = xGrid[j];
                zRow[j] = 0;
                cRow[j] = rawRow ? rawRow[c] : 128;
            }
            zGrid[i] = zRow;
            colorGrid[i] = cRow;
        }
    }

    const traces = [];

    // TRACE 1: Base 3D Surface (False-color HSI or Predicted Classification Map)
    const surfaceTrace = {
        type: 'surface',
        x: xGrid,
        y: yGrid,
        z: zGrid,
        surfacecolor: colorGrid,
        colorscale: colorscale,
        cmin: cmin,
        cmax: cmax,
        showscale: false,
        name: isPredictMode ? 'Predicted Classification Map' : `${currentDataset} Spatial Scene`,
        hoverinfo: 'none'
    };
    traces.push(surfaceTrace);

    // TRACE 2: Verified Human-Labelled Points Layer (solid circles with crisp white border at Z=1.0)
    if (show3DHuman && lastPredictionsData && lastPredictionsData.human_points && lastPredictionsData.human_points.length > 0) {
        const humanPts = lastPredictionsData.human_points;
        const hX = humanPts.map(p => p.col);
        const hY = humanPts.map(p => p.row);
        const hZ = new Array(humanPts.length).fill(1.0);
        const hColors = humanPts.map(p => {
            if (sceneData.class_colors && p.class_id >= 0 && p.class_id < sceneData.class_colors.length) {
                return sceneData.class_colors[p.class_id];
            }
            return '#3b82f6';
        });
        const hCustom = humanPts.map((p, idx) => [
            p.class_name,
            idx + 1,
            p.global_index,
            p.col,
            p.row
        ]);

        const humanTrace = {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Human-Labelled Points',
            x: hX,
            y: hY,
            z: hZ,
            marker: {
                size: 5.5,
                color: hColors,
                symbol: 'circle',
                line: { color: '#ffffff', width: 2.0 },
                opacity: 0.95
            },
            customdata: hCustom,
            hovertemplate:
                '<b>✓ Human-Labelled (Oracle Ground Truth)</b><br>' +
                'Class: <b>%{customdata[0]}</b><br>' +
                'Coordinates: (X: %{customdata[3]}, Y: %{customdata[4]})<br>' +
                'Verified Sample #%{customdata[1]} (Global: %{customdata[2]})<extra></extra>'
        };
        traces.push(humanTrace);
    }

    // TRACE 3: Accepted High-Confidence Pseudo-Labelled Points Layer (gold diamonds at Z=2.0 with stems)
    if (show3DPseudo && lastPredictionsData && lastPredictionsData.pseudo_points && lastPredictionsData.pseudo_points.length > 0) {
        const pseudoPts = lastPredictionsData.pseudo_points;
        const pX = pseudoPts.map(p => p.col);
        const pY = pseudoPts.map(p => p.row);
        const pZ = new Array(pseudoPts.length).fill(2.0);
        const pCustom = pseudoPts.map((p, idx) => [
            p.class_name,
            p.confidence,
            idx + 1,
            p.col,
            p.row
        ]);

        const pLineX = [];
        const pLineY = [];
        const pLineZ = [];
        for (let i = 0; i < pseudoPts.length; i++) {
            pLineX.push(pseudoPts[i].col, pseudoPts[i].col, null);
            pLineY.push(pseudoPts[i].row, pseudoPts[i].row, null);
            pLineZ.push(0, 2.0, null);
        }

        const pseudoLinesTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: pLineX,
            y: pLineY,
            z: pLineZ,
            line: {
                color: 'rgba(251, 191, 36, 0.40)',
                width: 1.5,
                dash: 'dot'
            },
            name: 'Pseudo-Label Projections',
            hoverinfo: 'none'
        };
        traces.push(pseudoLinesTrace);

        const pseudoTrace = {
            type: 'scatter3d',
            mode: 'markers',
            name: 'Accepted Pseudo-Labels (τ ≥ 0.95)',
            x: pX,
            y: pY,
            z: pZ,
            marker: {
                size: 6.5,
                color: '#fbbf24',
                symbol: 'diamond',
                line: { color: '#f59e0b', width: 2.0 },
                opacity: 0.95
            },
            customdata: pCustom,
            hovertemplate:
                '<b>⚡ Accepted Pseudo-Label (τ ≥ 0.95, λ=0.2)</b><br>' +
                'Predicted Class: <b>%{customdata[0]}</b><br>' +
                'Model Confidence: <b>%{customdata[1]:.4f}</b><br>' +
                'Coordinates: (X: %{customdata[3]}, Y: %{customdata[4]})<extra></extra>'
        };
        traces.push(pseudoTrace);
    }

    // TRACE 4: Queried Candidates Trace (at Z=4.0 with drop lines)
    const hasSamples = querySelectionData && querySelectionData.selected_samples && querySelectionData.selected_samples.length > 0;
    if (hasSamples && show3DQueried) {
        const samples = querySelectionData.selected_samples;
        const xCoords = samples.map(s => s.coord[1]);
        const yCoords = samples.map(s => s.coord[0]);
        const zCoords = new Array(samples.length).fill(4.0);
        const textLabels = samples.map(s => `#${s.rank}`);

        const stratName = currentStrategy === 'filter_diversity'
            ? 'LC + Farthest Diversity'
            : currentStrategy === 'least_confidence'
                ? 'Least Confidence'
                : 'Random Sampling';

        const customData = samples.map(s => [
            s.rank,
            s.global_index,
            s.coord[1],
            s.coord[0],
            s.least_confidence,
            s.predicted_class,
            stratName,
            s.oracle_class || 'Oracle Verified'
        ]);

        const qLineX = [];
        const qLineY = [];
        const qLineZ = [];
        for (let i = 0; i < samples.length; i++) {
            qLineX.push(samples[i].coord[1], samples[i].coord[1], null);
            qLineY.push(samples[i].coord[0], samples[i].coord[0], null);
            qLineZ.push(0, 4.0, null);
        }

        const dropLinesTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: qLineX,
            y: qLineY,
            z: qLineZ,
            line: {
                color: 'rgba(239, 68, 68, 0.45)',
                width: 1.5,
                dash: 'dash'
            },
            name: 'Queried Drop Lines',
            hoverinfo: 'none'
        };
        traces.push(dropLinesTrace);

        const queriedTrace = {
            type: 'scatter3d',
            mode: 'markers+text',
            name: 'Queried Candidates (#1–#20)',
            x: xCoords,
            y: yCoords,
            z: zCoords,
            text: textLabels,
            textposition: 'top center',
            textfont: {
                family: 'Inter, sans-serif',
                size: 11,
                color: '#ffffff',
                weight: 'bold'
            },
            marker: {
                size: 8,
                color: '#ef4444',
                symbol: 'diamond',
                line: { color: '#ffffff', width: 2 }
            },
            customdata: customData,
            hovertemplate:
                '<b>Queried Candidate #%{customdata[0]}</b><br>' +
                'Coordinates: (X: %{customdata[2]}, Y: %{customdata[3]})<br>' +
                'Uncertainty (LC): %{customdata[4]:.3f}<br>' +
                'Predicted: %{customdata[5]}<br>' +
                'Oracle Class: <b>%{customdata[7]}</b><br>' +
                'Selection Strategy: %{customdata[6]}<extra></extra>'
        };
        traces.push(queriedTrace);
    }

    // Preserve existing camera orientation safely without calling non-existent functions
    let camera = {
        eye: { x: 0.0, y: -1.35, z: 1.35 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 }
    };
    try {
        if (sceneEl._fullLayout && sceneEl._fullLayout.scene && sceneEl._fullLayout.scene.camera) {
            const curCam = sceneEl._fullLayout.scene.camera;
            if (curCam && curCam.eye) {
                camera = {
                    eye: { x: curCam.eye.x, y: curCam.eye.y, z: curCam.eye.z },
                    center: curCam.center ? { x: curCam.center.x, y: curCam.center.y, z: curCam.center.z } : { x: 0, y: 0, z: 0 },
                    up: curCam.up ? { x: curCam.up.x, y: curCam.up.y, z: curCam.up.z } : { x: 0, y: 0, z: 1 }
                };
            }
        }
    } catch (camErr) {
        console.warn("Could not read existing Plotly camera:", camErr);
    }

    const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            xaxis: {
                title: { text: 'Image Column (X)', font: { size: 10, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.15)',
                backgroundcolor: 'rgba(15, 23, 42, 0.5)',
                showbackground: true,
                range: [-2, width + 2]
            },
            yaxis: {
                title: { text: 'Image Row (Y)', font: { size: 10, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.15)',
                backgroundcolor: 'rgba(15, 23, 42, 0.5)',
                showbackground: true,
                autorange: 'reversed',
                range: [height + 2, -2]
            },
            zaxis: {
                title: { text: 'Visual Layers (Z)', font: { size: 9, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                backgroundcolor: 'rgba(15, 23, 42, 0.5)',
                showbackground: true,
                range: [-0.5, 6]
            },
            camera: camera,
            aspectratio: { x: 1.0, y: height / width, z: 0.28 }
        },
        showlegend: false
    };

    Plotly.react(sceneEl, traces, layout, { responsive: true, displaylogo: false }).then(() => {
        if (sceneEl.removeAllListeners) {
            sceneEl.removeAllListeners('plotly_click');
            sceneEl.removeAllListeners('plotly_hover');
        }

        sceneEl.on('plotly_click', (evt) => {
            if (evt && evt.points && evt.points.length > 0) {
                const pt = evt.points[0];
                if (pt.customdata) {
                    if (pt.data.name && pt.data.name.includes('Queried')) {
                        const [rank, gIdx, col, row, lc, predCls, strat, oraCls] = pt.customdata;
                        updateScene3DInfoCard(rank, col, row, lc, predCls, strat, gIdx, oraCls);
                        inspectSample(gIdx, rank);
                    } else if (pt.data.name && pt.data.name.includes('Human')) {
                        const [cName, sampleNum, gIdx, col, row] = pt.customdata;
                        showGeneric3DInfo(`<b>✓ Verified Human-Labelled #${sampleNum}</b> &bull; Class: <strong style="color:var(--primary);">${cName}</strong> &bull; Coord: (${col}, ${row}) &bull; Global Index: ${gIdx}`);
                    } else if (pt.data.name && pt.data.name.includes('Pseudo')) {
                        const [cName, conf, pNum, col, row] = pt.customdata;
                        showGeneric3DInfo(`<b>⚡ Accepted Pseudo-Label #${pNum} (τ ≥ 0.95)</b> &bull; Predicted: <strong style="color:#fbbf24;">${cName}</strong> &bull; Confidence: <strong>${conf.toFixed(4)}</strong> &bull; Coord: (${col}, ${row})`);
                    }
                }
            }
        });

        sceneEl.on('plotly_hover', (evt) => {
            if (evt && evt.points && evt.points.length > 0) {
                const pt = evt.points[0];
                if (pt.customdata) {
                    if (pt.data.name && pt.data.name.includes('Queried')) {
                        const [rank, gIdx, col, row, lc, predCls, strat, oraCls] = pt.customdata;
                        updateScene3DInfoCard(rank, col, row, lc, predCls, strat, gIdx, oraCls);
                    } else if (pt.data.name && pt.data.name.includes('Human')) {
                        const [cName, sampleNum, gIdx, col, row] = pt.customdata;
                        showGeneric3DInfo(`<b>✓ Verified Human-Labelled #${sampleNum}</b> &bull; Class: <strong style="color:var(--primary);">${cName}</strong> &bull; Coord: (${col}, ${row}) &bull; Global Index: ${gIdx}`);
                    } else if (pt.data.name && pt.data.name.includes('Pseudo')) {
                        const [cName, conf, pNum, col, row] = pt.customdata;
                        showGeneric3DInfo(`<b>⚡ Accepted Pseudo-Label #${pNum} (τ ≥ 0.95)</b> &bull; Predicted: <strong style="color:#fbbf24;">${cName}</strong> &bull; Confidence: <strong>${conf.toFixed(4)}</strong> &bull; Coord: (${col}, ${row})`);
                    }
                }
            }
        });
    }).catch(err => {
        console.error("Plotly.react render error in 3D scene:", err);
    });
}
function updateScene3DInfoCard(rank, x, y, unc, predCls, strat, gIdx, oraCls) {
    const infoEl = document.getElementById("scene3dSelectedInfo");
    if (!infoEl) return;
    const uncStr = typeof unc === 'number' ? unc.toFixed(3) : unc;
    const oracleSnippet = oraCls ? `<strong style="color:var(--text-secondary); margin-left:6px;">Oracle:</strong> <span style="color:var(--success); font-weight:700;">${oraCls}</span> &bull;` : '';
    infoEl.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; width:100%;">
            <div style="font-size:11.5px; line-height:1.5;">
                <span class="badge badge-warning" style="font-weight:700;">Selected Sample #${rank}</span>
                <strong style="margin-left:8px; color:var(--text-primary);">Position:</strong> (x=${x}, y=${y}) &bull;
                <strong style="color:var(--text-secondary);">Uncertainty (LC):</strong> <span style="color:var(--danger); font-weight:700;">${uncStr}</span> &bull;
                <strong style="color:var(--text-secondary);">Predicted class:</strong> <span style="color:var(--primary); font-weight:600;">${predCls}</span> &bull;
                ${oracleSnippet}
                <strong style="color:var(--text-secondary);">Selection method:</strong> <span>${strat}</span>
            </div>
            <button class="btn btn-primary btn-sm" onclick="inspectSample(${gIdx}, ${rank})" style="font-size:11px; padding:3px 10px;">🔍 Inspect Sample #${rank}</button>
        </div>
    `;
}

function resetScene3DCamera() {
    stopScene3DAutoRotate();
    const sceneEl = document.getElementById("hsiSpatialScene3D");
    if (sceneEl && window.Plotly) {
        Plotly.relayout(sceneEl, {
            'scene.camera': {
                eye: { x: 0.0, y: -1.35, z: 1.35 },
                center: { x: 0, y: 0, z: 0 },
                up: { x: 0, y: 0, z: 1 }
            }
        });
    }
}

function toggleScene3DSamples() {
    show3DSamples = !show3DSamples;
    const btn = document.getElementById("btnToggle3DSamples");
    if (btn) {
        btn.innerText = show3DSamples ? "👁 Toggle Markers" : "👁 Show Markers";
        btn.style.opacity = show3DSamples ? "1" : "0.7";
    }
    renderScene3D();
}

function toggleScene3DAutoRotate() {
    if (isScene3DAutoRotating) {
        stopScene3DAutoRotate();
    } else {
        startScene3DAutoRotate();
    }
}

function startScene3DAutoRotate() {
    const sceneEl = document.getElementById("hsiSpatialScene3D");
    if (!sceneEl || !window.Plotly) return;

    isScene3DAutoRotating = true;
    const btn = document.getElementById("btnScene3DAutoRotate");
    const icon = document.getElementById("scene3dRotateIcon");
    if (btn) btn.classList.add("active");
    if (icon) icon.innerText = "⏸";

    scene3DCameraAngle = -Math.PI / 2;
    const radius = 1.9;
    const eyeZ = 1.35;

    function step() {
        if (!isScene3DAutoRotating) return;
        scene3DCameraAngle += 0.012;
        const eyeX = radius * Math.cos(scene3DCameraAngle);
        const eyeY = radius * Math.sin(scene3DCameraAngle);

        Plotly.relayout(sceneEl, {
            'scene.camera.eye': { x: eyeX, y: eyeY, z: eyeZ }
        }).then(() => {
            if (isScene3DAutoRotating) {
                scene3DRotationRafId = requestAnimationFrame(step);
            }
        }).catch(() => stopScene3DAutoRotate());
    }

    scene3DRotationRafId = requestAnimationFrame(step);
}

function stopScene3DAutoRotate() {
    isScene3DAutoRotating = false;
    if (scene3DRotationRafId) {
        cancelAnimationFrame(scene3DRotationRafId);
        scene3DRotationRafId = null;
    }
    const btn = document.getElementById("btnScene3DAutoRotate");
    const icon = document.getElementById("scene3dRotateIcon");
    if (btn) btn.classList.remove("active");
    if (icon) icon.innerText = "⟳";
}

// DEVELOPER AUDIT VERIFICATION LOGGING
function logHsiSampleDebug(data) {
    if (!data) return;
    const totalBands = data.spectrum ? data.spectrum.length : (data.patch_hsi ? data.patch_hsi.length : 0);
    const maxProb = data.probabilities ? Math.max(...data.probabilities) : 0;
    const predConf = (maxProb * 100).toFixed(2) + "%";
    const unc = typeof data.least_confidence === "number" ? data.least_confidence.toFixed(4) : "N/A";

    const isHuman = (data.mode === "human" || currentMode === "human");
    const gtInfo = isHuman ? "[REDACTED - Human Annotation Privacy Active]" : (data.gt_class || "N/A");

    const debugEl = document.getElementById("hsiDebugCard");
    if (debugEl) {
        debugEl.innerHTML = `
            <div style="font-family:monospace; font-size:11px; line-height:1.6; color:#94a3b8;">
                <div><span style="color:#38bdf8;">Dataset:</span> <span style="color:#f8fafc; font-weight:600;">${currentDataset}</span> | <span style="color:#38bdf8;">Sample index:</span> <span style="color:#f8fafc; font-weight:600;">${data.global_index}</span> | <span style="color:#38bdf8;">Patch shape:</span> <span style="color:#f8fafc;">15&times;15</span></div>
                <div><span style="color:#38bdf8;">Spatial Coords:</span> <span style="color:#f8fafc; font-weight:600;">(x=${data.coord ? data.coord[1] : 'N/A'}, y=${data.coord ? data.coord[0] : 'N/A'})</span> | <span style="color:#38bdf8;">Bands:</span> <span style="color:#f8fafc; font-weight:600;">${totalBands}</span></div>
                <div><span style="color:#38bdf8;">Prediction:</span> <span style="color:#f8fafc; font-weight:600;">${data.predicted_class}</span> (conf: <span style="color:#f8fafc;">${predConf}</span>) | <span style="color:#38bdf8;">Uncertainty (LC):</span> <span style="color:#f8fafc;">${unc}</span></div>
                <div><span style="color:#38bdf8;">Human Mode GT Privacy:</span> <span style="color:#38bdf8; font-weight:600;">${gtInfo}</span></div>
            </div>
        `;
    }
}

// Window resize listener
window.addEventListener("resize", () => {
    if (currentSceneViewMode === '3d' && window.Plotly) {
        const sceneEl = document.getElementById("hsiSpatialScene3D");
        if (sceneEl) Plotly.Plots.resize(sceneEl);
    }
});

// Global modal cleanup on Escape key
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        const modal = document.getElementById("sampleModal");
        if (modal && modal.classList.contains("active")) {
            modal.classList.remove("active");
        }
    }
});

window.renderScene3D = renderScene3D;
window.setSceneViewMode = setSceneViewMode;
window.resetScene3DCamera = resetScene3DCamera;
window.toggleScene3DAutoRotate = toggleScene3DAutoRotate;
window.toggleScene3DSamples = toggleScene3DSamples;
window.inspectSample = inspectSample;
window.update3DLayers = update3DLayers;
window.autoFillOracleLabels = autoFillOracleLabels;
window.update3DStatsPanel = update3DStatsPanel;

window.switchDatasetExplorerView = switchDatasetExplorerView;
window.loadBenchmarkData = loadBenchmarkData;
window.renderBenchmarkTab = renderBenchmarkTab;
window.loadCrossSensorPair = loadCrossSensorPair;
window.loadCrossSensorData = loadCrossSensorData;
window.runLiveAdaptationTrial = runLiveAdaptationTrial;
window.switchLabelEffProtocol = switchLabelEffProtocol;
window.loadClassificationResults = loadClassificationResults;
window.calculateLabelCost = calculateLabelCost;


