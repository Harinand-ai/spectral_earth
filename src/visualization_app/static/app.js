// GLOBAL APPLICATION STATE
let currentDataset = "Indian Pines";
let currentStrategy = "filter_diversity";
let currentMode = "oracle"; // "oracle" or "human"
let targetOA = 70.0;
let sceneData = null;
let currentMapType = "rgb"; // "rgb", "pool", "pred", "unc"
let currentUncertaintyMethod = "least_confidence";
let lastPredictionsData = null;
let querySelectionData = null;
let isDatasetLoading = false;
let hasPredicted = false;
let hasQueried = false;

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
    hasQueried = false;
    querySelectionData = null;

    const sceneEmpty = document.getElementById("sceneEmptyState");
    const sceneActive = document.getElementById("sceneActiveContent");
    const sceneControls = document.getElementById("sceneMapControls");
    if (sceneEmpty) sceneEmpty.style.display = "flex";
    if (sceneActive) sceneActive.style.display = "none";
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

    const btnPredict = document.getElementById("btnInitialPredict");
    if (btnPredict) {
        btnPredict.disabled = false;
        btnPredict.className = "btn btn-primary";
        btnPredict.innerHTML = "⚡ Initial Prediction";
    }

    const btnQuery = document.getElementById("btnRunQuery");
    if (btnQuery) {
        btnQuery.disabled = true;
        btnQuery.className = "btn btn-secondary disabled-stage";
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
    sampleSpectrum: '2d'
};
let lastProgressionHistory = null;
let lastResamplerData = null;
let lastBenchmarkParams = { dsName: 'Indian Pines', metricView: 'oa' };
let lastLabelEfficiencyCurves = null;
let lastSampleData = null;

function setChartMode(chartName, mode) {
    chartModes[chartName] = mode;
    const btn3d = document.getElementById(`btn${capitalizeFirst(chartName)}3D`);
    const btn2d = document.getElementById(`btn${capitalizeFirst(chartName)}2D`);
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

document.addEventListener("DOMContentLoaded", () => {
    initTabNavigation();
    initEventListeners();
    initGuidedEngine();
    loadDataset(currentDataset, false);
    loadCrossSensorData("Indian Pines -> Pavia University", false);
    updateExperimentalEvidenceView();
});

/* =========================================================================
   NAVIGATION & TABS
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

            // Auto-resize Plotly 3D graphs when tab becomes visible
            setTimeout(() => {
                ['progressionChart', 'resamplerChart', 'benchmarkChart', 'labelEfficiencyChart'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el && window.Plotly && el.offsetHeight > 0) {
                        Plotly.Plots.resize(el);
                    }
                });
            }, 60);
        });
    });
}

/* =========================================================================
   EVENT LISTENERS (SINGLE AUTHORITATIVE HANDLERS)
   ========================================================================= */
function initEventListeners() {
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
                if (sceneEmpty) sceneEmpty.style.display = "none";
                if (sceneActive) sceneActive.style.display = "block";
                setSceneViewMode(currentSceneViewMode);

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
                btnPredict.innerHTML = "Initial Prediction";
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

    // Cross-Sensor Transfer Pair Selector
    const csPair = document.getElementById("transferPairSelect");
    if (csPair) {
        csPair.addEventListener("change", (e) => {
            loadCrossSensorData(e.target.value, true);
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

        // Progressive disclosure initial state
        resetProgressiveDisclosure();
        await fetchRoundProgression();
        isDatasetLoading = false;

        // Render clean 3D scene plane for newly selected dataset
        renderScene3D();

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

    document.querySelectorAll(".segmented-btn").forEach(b => b.classList.remove("active"));
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
                    const [red, green, blue] = hexToRgb(hex);
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

function hexToRgb(hex) {
    let c = hex.replace("#", "");
    if (c.length === 3) c = c.split("").map(x => x + x).join("");
    const num = parseInt(c, 16);
    return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
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

        renderQuerySelectionTable();
        renderSpatialCanvas();
        renderScene3D();

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

    querySelectionData.selected_samples.forEach((s) => {
        const tr = document.createElement("tr");

        let classSelectHtml = "";
        if (sceneData && sceneData.class_names) {
            const options = sceneData.class_names.map(cName => {
                const selected = cName === s.predicted_class ? "selected" : "";
                return `<option value="${cName}" ${selected}>${cName}</option>`;
            }).join("");
            classSelectHtml = `<select class="select-input user-class-select" data-global-index="${s.global_index}" style="padding:2px 6px; font-size:11px;">${options}</select>`;
        } else {
            classSelectHtml = s.predicted_class;
        }

        tr.innerHTML = `
            <td><strong>#${s.rank}</strong></td>
            <td>(${s.coord[0]}, ${s.coord[1]})</td>
            <td><span style="display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px; background:${getClassColor(s.predicted_class)}; color:#ffffff; font-weight:600;">${s.predicted_class}</span></td>
            <td><strong style="color:var(--danger);">${s.least_confidence.toFixed(3)}</strong></td>
            <td style="font-size:11px; color:var(--text-secondary); max-width:180px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;" title="${s.selection_reason}">${s.selection_reason}</td>
            <td><button class="btn btn-secondary btn-sm" onclick="inspectSample(${s.global_index}, ${s.rank})">Inspect</button></td>
            <td>${currentMode === "human" ? classSelectHtml : `<span style="font-size:11px; color:var(--text-muted);">Oracle Ground Truth</span>`}</td>
        `;
        tbody.appendChild(tr);
    });
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
            body: JSON.stringify({ epochs: 25, lr: 0.0005 }),
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

        // Update 3D scene & 2D canvas
        renderSpatialCanvas();
        renderScene3D();
        const badgeEl = document.getElementById("scene3dSamplesBadge");
        if (badgeEl) badgeEl.innerHTML = `Query Informative Samples to display 20 AL candidate locations in 3D`;
        const infoEl = document.getElementById("scene3dSelectedInfo");
        if (infoEl) infoEl.innerHTML = `<span style="color:var(--success); font-weight:600; margin-right:6px;">✓</span> Retraining complete. Model updated with newly labeled samples.`;

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

        const res = await fetch("/api/calculate_cost", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cost_per_label: costPerLabel, dataset_name: currentDataset }),
        });
        const data = await res.json();

        if (document.getElementById("costRandom")) document.getElementById("costRandom").innerText = `₹${data.random.cost.toLocaleString()}`;
        if (document.getElementById("labelsRandom")) document.getElementById("labelsRandom").innerText = `${data.random.labels} labels required`;

        if (document.getElementById("costLC")) document.getElementById("costLC").innerText = `₹${data.least_confidence.cost.toLocaleString()}`;
        if (document.getElementById("labelsLC")) document.getElementById("labelsLC").innerText = `${data.least_confidence.labels} labels required`;

        if (document.getElementById("costLCFD")) document.getElementById("costLCFD").innerText = `₹${data.lc_fd.cost.toLocaleString()}`;
        if (document.getElementById("labelsLCFD")) document.getElementById("labelsLCFD").innerText = `${data.lc_fd.labels} labels required`;

        if (document.getElementById("costSaved")) document.getElementById("costSaved").innerText = `₹${data.savings.cost_saved.toLocaleString()} (${data.savings.percentage_workload_reduction}%)`;
        if (document.getElementById("labelsSaved")) document.getElementById("labelsSaved").innerText = `Saved ${data.savings.labels_saved} label annotations`;

        if (document.getElementById("costDisclaimer")) document.getElementById("costDisclaimer").innerText = data.disclaimer;
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

        renderProgressionChart(history);
    } catch (err) {
        console.error("Error fetching progression:", err);
    }
}

function renderProgressionChart(history) {
    const chartEl = document.getElementById("progressionChart");
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
async function loadCrossSensorData(pairKey, userInitiated = false) {
    try {
        const res = await fetch(`/api/cross_sensor_data?pair_key=${encodeURIComponent(pairKey)}`);
        const data = await res.json();

        const summary = data.summary;
        const meta = summary.metadata;

        if (document.getElementById("csSource")) document.getElementById("csSource").innerText = meta.source_sensor;
        if (document.getElementById("csTarget")) document.getElementById("csTarget").innerText = meta.target_sensor;

        let zeroShotVal = "N/A";
        if (pairKey.includes("Indian Pines -> Salinas")) zeroShotVal = "15.25%";
        else if (pairKey.includes("Salinas -> Indian Pines")) zeroShotVal = "5.59%";
        else if (summary.zero_shot_metrics && summary.zero_shot_metrics.oa) zeroShotVal = `${summary.zero_shot_metrics.oa.toFixed(2)}%`;

        if (document.getElementById("csZeroShot")) document.getElementById("csZeroShot").innerText = zeroShotVal;

        let fewShotVal = "49.84%";
        if (summary.few_shot_final_metrics && summary.few_shot_final_metrics.oa) fewShotVal = `${summary.few_shot_final_metrics.oa.toFixed(2)}%`;
        if (document.getElementById("csFewShot")) document.getElementById("csFewShot").innerText = fewShotVal;

        renderCrossSensorRoundTable(summary.round_progression);
        render1DSpectralResamplerChart(data.resampler_data);

        // Guided State Machine Event
        if (window.tutorialEngine && window.tutorialEngine.isActive) {
            window.tutorialEngine.notifyAction("CROSS_SENSOR_DONE", {
                pairKey,
                fewShotVal,
                sourceSensor: meta.source_sensor,
                targetSensor: meta.target_sensor
            });
        }
    } catch (err) {
        console.error("Error loading cross-sensor data:", err);
    }
}

function renderCrossSensorRoundTable(roundProgression) {
    const tbody = document.querySelector("#csRoundTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    roundProgression.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>Round ${r.round}</td>
            <td>${r.num_labeled}</td>
            <td><strong>${r.oa.toFixed(2)}%</strong></td>
            <td>${r.f1.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function render1DSpectralResamplerChart(resamplerData) {
    const chartEl = document.getElementById("resamplerChart");
    if (!chartEl) return;
    lastResamplerData = resamplerData;

    const mode = chartModes.resampler || '3d';
    const is3D = (mode === '3d');

    const waves = resamplerData.source_wavelengths.map(w => Math.round(w));
    const targetSpec = resamplerData.target_spectrum;
    const resampledSpec = resamplerData.resampled_spectrum;

    if (is3D && window.Plotly) {
        // Trace 0: Target Grid (Red line at Y=0)
        const traceTarget = {
            type: 'scatter3d',
            mode: 'lines',
            name: `Target Grid (${resamplerData.target_bands_count} Bands)`,
            x: waves,
            y: Array(waves.length).fill(0),
            z: targetSpec,
            line: { color: '#ef4444', width: 6 },
            hovertemplate: '<b>Target Sensor Grid</b><br>Wavelength: %{x} nm<br>Reflectance: %{z:.4f}<extra></extra>'
        };

        // Trace 1: 1D Resampled (Blue line at Y=1)
        const traceResampled = {
            type: 'scatter3d',
            mode: 'lines',
            name: `1D Resampled (${resamplerData.source_bands_count} Bands)`,
            x: waves,
            y: Array(waves.length).fill(1),
            z: resampledSpec,
            line: { color: '#2563eb', width: 6 },
            hovertemplate: '<b>1D Resampled Profile</b><br>Wavelength: %{x} nm<br>Reflectance: %{z:.4f}<extra></extra>'
        };

        // 3D Connecting Spectral Transfer Surface Mesh between Target and Resampled
        const N = waves.length;
        const meshX = [...waves, ...waves];
        const meshY = [...Array(N).fill(0), ...Array(N).fill(1)];
        const meshZ = [...targetSpec, ...resampledSpec];
        const iIdx = [];
        const jIdx = [];
        const kIdx = [];
        for (let idx = 0; idx < N - 1; idx++) {
            iIdx.push(idx);
            jIdx.push(idx + 1);
            kIdx.push(idx + N);

            iIdx.push(idx + 1);
            jIdx.push(idx + N + 1);
            kIdx.push(idx + N);
        }

        const traceMesh = {
            type: 'mesh3d',
            name: 'Spectral Transfer Ribbon',
            x: meshX,
            y: meshY,
            z: meshZ,
            i: iIdx,
            j: jIdx,
            k: kIdx,
            opacity: 0.28,
            color: '#3b82f6',
            hoverinfo: 'name'
        };

        // Trace 3: Spectral Alignment Gap (Delta at Y = 0.5)
        const deltas = targetSpec.map((v, i) => Math.abs(v - (resampledSpec[i] || 0)));
        const traceDelta = {
            type: 'scatter3d',
            mode: 'lines',
            name: 'Residual Gap |Δ|',
            x: waves,
            y: Array(waves.length).fill(0.5),
            z: deltas,
            line: { color: '#f59e0b', width: 3, dash: 'dot' },
            hovertemplate: '<b>Residual Gap</b><br>Wavelength: %{x} nm<br>|Δ|: %{z:.4f}<extra></extra>'
        };

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: {
                    title: { text: 'Wavelength (nm)', font: { size: 10, color: '#64748b' } },
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                yaxis: {
                    title: { text: 'Domain Track', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 0.5, 1],
                    ticktext: ['Target Grid', '|Δ| Gap', '1D Resampled'],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                zaxis: {
                    title: { text: 'Reflectance', font: { size: 10, color: '#64748b' } },
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
                font: { size: 10 }
            }
        };

        Plotly.react(chartEl, [traceTarget, traceResampled, traceMesh, traceDelta], layout, { responsive: true, displaylogo: false });
    } else if (window.Plotly) {
        // 2D View
        const traceTarget = {
            type: 'scatter',
            mode: 'lines',
            name: `Target Grid (${resamplerData.target_bands_count} Bands)`,
            x: waves,
            y: targetSpec,
            line: { color: '#ef4444', width: 2 }
        };

        const traceResampled = {
            type: 'scatter',
            mode: 'lines',
            name: `1D Resampled (${resamplerData.source_bands_count} Bands)`,
            x: waves,
            y: resampledSpec,
            line: { color: '#2563eb', width: 2, dash: 'dash' }
        };

        const layout = {
            margin: { l: 45, r: 20, b: 35, t: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { title: 'Wavelength (nm)', gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            yaxis: { title: 'Reflectance', gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            legend: { x: 0.02, y: 0.98, font: { size: 10 } }
        };

        Plotly.react(chartEl, [traceTarget, traceResampled], layout, { responsive: true, displaylogo: false });
    }
}

/* =========================================================================
   CONNECTED EXPERIMENTAL EVIDENCE CONTROLLER
   ========================================================================= */
function updateExperimentalEvidenceView() {
    const dsName = document.getElementById("expDatasetSelect") ? document.getElementById("expDatasetSelect").value : "Indian Pines";
    const expType = document.getElementById("expTypeSelect") ? document.getElementById("expTypeSelect").value : "benchmark";
    const metricView = document.getElementById("expMetricSelect") ? document.getElementById("expMetricSelect").value : "all";

    const metricField = document.getElementById("metricControlField");
    if (metricField) {
        metricField.style.display = (expType === "benchmark" || expType === "efficiency") ? "block" : "none";
    }

    document.querySelectorAll(".exp-view").forEach(v => v.style.display = "none");

    if (expType === "benchmark") {
        const viewEl = document.getElementById("view-exp-benchmark");
        if (viewEl) viewEl.style.display = "block";

        renderBenchmarkTable(dsName, metricView);
        renderBenchmarkChart(dsName, metricView);

        const paviaBox = document.getElementById("paviaInterpretationBox");
        if (paviaBox) paviaBox.style.display = (dsName === "Pavia University") ? "block" : "none";

    } else if (expType === "efficiency") {
        const viewEl = document.getElementById("view-exp-efficiency");
        if (viewEl) viewEl.style.display = "block";

        loadLabelEfficiencyData(dsName);

    } else if (expType === "ablation") {
        const viewEl = document.getElementById("view-exp-ablation");
        if (viewEl) viewEl.style.display = "block";

    } else if (expType === "calibration") {
        const viewEl = document.getElementById("view-exp-calibration");
        if (viewEl) viewEl.style.display = "block";

    } else if (expType === "cost") {
        const viewEl = document.getElementById("view-exp-cost");
        if (viewEl) viewEl.style.display = "block";
    }
}

function renderBenchmarkTable(dsName, metricView) {
    const dataList = BENCHMARK_DATA[dsName] || BENCHMARK_DATA["Indian Pines"];
    const tbody = document.querySelector("#benchmarkTable tbody");
    const thead = document.querySelector("#benchmarkTable thead tr");
    if (!tbody || !thead) return;

    if (metricView === "oa") {
        thead.innerHTML = `<th>Method</th><th>Overall Accuracy (OA %)</th>`;
    } else if (metricView === "aa") {
        thead.innerHTML = `<th>Method</th><th>Average Accuracy (AA %)</th>`;
    } else if (metricView === "f1") {
        thead.innerHTML = `<th>Method</th><th>Macro F1 Score</th>`;
    } else if (metricView === "kappa") {
        thead.innerHTML = `<th>Method</th><th>Kappa Coefficient (&kappa;)</th>`;
    } else {
        thead.innerHTML = `<th>Method</th><th>Overall Accuracy (OA %)</th><th>Average Accuracy (AA %)</th><th>Macro F1</th><th>Kappa (&kappa;)</th>`;
    }

    tbody.innerHTML = "";

    dataList.forEach(row => {
        const tr = document.createElement("tr");
        if (row.isBest) tr.style.background = "#eff6ff";

        const mName = row.isBest ? `<strong>${row.method} (Best)</strong>` : row.method;

        if (metricView === "oa") {
            tr.innerHTML = `<td>${mName}</td><td><strong>${row.oa}</strong></td>`;
        } else if (metricView === "aa") {
            tr.innerHTML = `<td>${mName}</td><td><strong>${row.aa}</strong></td>`;
        } else if (metricView === "f1") {
            tr.innerHTML = `<td>${mName}</td><td><strong>${row.f1}</strong></td>`;
        } else if (metricView === "kappa") {
            tr.innerHTML = `<td>${mName}</td><td><strong>${row.kappa}</strong></td>`;
        } else {
            tr.innerHTML = `
                <td>${mName}</td>
                <td><strong>${row.oa}</strong></td>
                <td>${row.aa}</td>
                <td>${row.f1}</td>
                <td>${row.kappa}</td>
            `;
        }
        tbody.appendChild(tr);
    });
}

function renderBenchmarkChart(dsName, metricView) {
    const chartEl = document.getElementById("benchmarkChart");
    if (!chartEl) return;
    lastBenchmarkParams = { dsName, metricView };

    const dataList = BENCHMARK_DATA[dsName] || BENCHMARK_DATA["Indian Pines"];
    const mode = chartModes.benchmark || '3d';
    const is3D = (mode === '3d');

    if (is3D && window.Plotly) {
        // Multi-metric 3D comparative columns landscape
        const metrics = ['OA (%)', 'AA (%)', 'Macro F1', 'Kappa (κ)'];
        const colors = ['#64748b', '#f59e0b', '#2563eb'];
        const traces = [];

        dataList.forEach((d, mIdx) => {
            const oa = parseFloat(d.oa.split(" ")[0]);
            const aa = parseFloat(d.aa.split(" ")[0]);
            const f1 = parseFloat(d.f1.split(" ")[0]) * 100;
            const kappa = parseFloat(d.kappa.split(" ")[0]) * 100;
            const zVals = [oa, aa, f1, kappa];

            // 3D Stems and illuminated markers
            zVals.forEach((val, yIdx) => {
                traces.push({
                    type: 'scatter3d',
                    mode: 'lines',
                    showlegend: false,
                    x: [mIdx, mIdx],
                    y: [yIdx, yIdx],
                    z: [0, val],
                    line: { color: colors[mIdx], width: 10 },
                    hoverinfo: 'skip'
                });
            });

            // Top marker caps
            traces.push({
                type: 'scatter3d',
                mode: 'markers+text',
                name: d.method,
                x: [mIdx, mIdx, mIdx, mIdx],
                y: [0, 1, 2, 3],
                z: zVals,
                text: zVals.map(v => v.toFixed(1) + '%'),
                textposition: 'top center',
                textfont: { size: 10, color: colors[mIdx] },
                marker: { size: 8, color: colors[mIdx], symbol: 'diamond' },
                hovertemplate: `<b>${d.method}</b><br>Metric: %{customdata}<br>Score: %{z:.2f}%<extra></extra>`,
                customdata: metrics
            });
        });

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: {
                    title: { text: 'Method', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 1, 2],
                    ticktext: ['Random', 'Least Conf', 'LC+FD (SE)'],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                yaxis: {
                    title: { text: 'Metric', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 1, 2, 3],
                    ticktext: ['OA (%)', 'AA (%)', 'F1 (x100)', 'Kappa (x100)'],
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
                    eye: { x: 1.7, y: -1.7, z: 1.3 },
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

        Plotly.react(chartEl, traces, layout, { responsive: true, displaylogo: false });
    } else if (window.Plotly) {
        // 2D Bar Chart
        let chartTitle = "Overall Accuracy (OA %)";
        let values = dataList.map(d => parseFloat(d.oa.split(" ")[0]));
        if (metricView === "aa") {
            chartTitle = "Average Accuracy (AA %)";
            values = dataList.map(d => parseFloat(d.aa.split(" ")[0]));
        } else if (metricView === "f1") {
            chartTitle = "Macro F1 Score (x100)";
            values = dataList.map(d => parseFloat(d.f1.split(" ")[0]) * 100);
        } else if (metricView === "kappa") {
            chartTitle = "Kappa Coefficient (x100)";
            values = dataList.map(d => parseFloat(d.kappa.split(" ")[0]) * 100);
        }

        const trace = {
            type: 'bar',
            x: dataList.map(d => d.method),
            y: values,
            marker: { color: ["#64748b", "#f59e0b", "#2563eb"], borderRadius: 4 },
            text: values.map(v => v.toFixed(2) + '%'),
            textposition: 'auto'
        };

        const layout = {
            title: { text: `${dsName} — ${chartTitle}`, font: { size: 12, color: '#1e293b' } },
            margin: { l: 45, r: 20, b: 45, t: 35 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            yaxis: { title: 'Score (%)', range: [0, 100], gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } }
        };

        Plotly.react(chartEl, [trace], layout, { responsive: true, displaylogo: false });
    }
}

/* =========================================================================
   LABEL-EFFICIENCY BENCHMARK & THRESHOLDS
   ========================================================================= */
async function loadLabelEfficiencyData(dsName = "Indian Pines") {
    try {
        const res = await fetch("/api/label_efficiency_data");
        const data = await res.json();

        const selectedDsCurves = data.curves_data[dsName] || data.curves_data["Indian Pines"];
        renderLabelEfficiencyChart(selectedDsCurves);

        const filteredThresholds = data.thresholds_table.filter(t => t.Dataset === dsName);
        renderThresholdsTable(filteredThresholds.length > 0 ? filteredThresholds : data.thresholds_table);
    } catch (err) {
        console.error("Error loading label-efficiency data:", err);
    }
}

function renderLabelEfficiencyChart(dsCurves) {
    const chartEl = document.getElementById("labelEfficiencyChart");
    if (!chartEl) return;
    lastLabelEfficiencyCurves = dsCurves;

    const mode = chartModes.labelEff || '3d';
    const is3D = (mode === '3d');

    const counts = dsCurves.counts;
    const randomVals = dsCurves["Random"];
    const lcVals = dsCurves["Least Confidence"];
    const lcfdVals = dsCurves["LC + Farthest Diversity"];

    if (is3D && window.Plotly) {
        // 3D Efficiency Trajectories in 3D Space
        const traceRandom = {
            type: 'scatter3d',
            mode: 'lines+markers',
            name: 'Random Sampling',
            x: counts,
            y: Array(counts.length).fill(0),
            z: randomVals,
            line: { color: '#64748b', width: 5 },
            marker: { size: 5, color: '#94a3b8' },
            hovertemplate: '<b>Random Sampling</b><br>Labels: %{x}<br>OA: %{z:.2f}%<extra></extra>'
        };

        const traceLC = {
            type: 'scatter3d',
            mode: 'lines+markers',
            name: 'Least Confidence',
            x: counts,
            y: Array(counts.length).fill(1),
            z: lcVals,
            line: { color: '#f59e0b', width: 5 },
            marker: { size: 5, color: '#fbbf24' },
            hovertemplate: '<b>Least Confidence</b><br>Labels: %{x}<br>OA: %{z:.2f}%<extra></extra>'
        };

        const traceLCFD = {
            type: 'scatter3d',
            mode: 'lines+markers',
            name: 'LC + Farthest Diversity',
            x: counts,
            y: Array(counts.length).fill(2),
            z: lcfdVals,
            line: { color: '#2563eb', width: 7 },
            marker: { size: 6, color: '#1d4ed8' },
            hovertemplate: '<b>LC + FD</b><br>Labels: %{x}<br>OA: %{z:.2f}%<extra></extra>'
        };

        // 3D Target threshold plane at 70%
        const minX = counts[0];
        const maxX = counts[counts.length - 1];
        const targetVal = 70.0;
        const traceTarget = {
            type: 'mesh3d',
            name: `Target OA (${targetVal}%)`,
            x: [minX, maxX, maxX, minX],
            y: [-0.3, -0.3, 2.3, 2.3],
            z: [targetVal, targetVal, targetVal, targetVal],
            i: [0, 0],
            j: [1, 2],
            k: [2, 3],
            opacity: 0.18,
            color: '#10b981',
            hoverinfo: 'name'
        };

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: {
                    title: { text: 'Labeled Count', font: { size: 10, color: '#64748b' } },
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                yaxis: {
                    title: { text: 'Acquisition Strategy', font: { size: 10, color: '#64748b' } },
                    tickvals: [0, 1, 2],
                    ticktext: ['Random', 'Least Conf', 'LC+FD'],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                zaxis: {
                    title: { text: 'Accuracy (%)', font: { size: 10, color: '#64748b' } },
                    range: [30, 85],
                    gridcolor: '#e2e8f0',
                    backgroundcolor: 'rgba(248, 250, 252, 0.5)',
                    showbackground: true
                },
                camera: {
                    eye: { x: 1.6, y: -1.7, z: 1.2 },
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

        Plotly.react(chartEl, [traceRandom, traceLC, traceLCFD, traceTarget], layout, { responsive: true, displaylogo: false });
    } else if (window.Plotly) {
        // 2D View
        const traceRandom = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Random Sampling',
            x: counts,
            y: randomVals,
            line: { color: '#64748b', width: 2 },
            marker: { size: 6 }
        };

        const traceLC = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Least Confidence',
            x: counts,
            y: lcVals,
            line: { color: '#f59e0b', width: 2 },
            marker: { size: 6 }
        };

        const traceLCFD = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'LC + Farthest Diversity',
            x: counts,
            y: lcfdVals,
            line: { color: '#2563eb', width: 3 },
            marker: { size: 7 }
        };

        const traceTarget = {
            type: 'scatter',
            mode: 'lines',
            name: 'Target (70%)',
            x: counts,
            y: Array(counts.length).fill(70),
            line: { color: '#10b981', width: 1.5, dash: 'dot' }
        };

        const layout = {
            margin: { l: 45, r: 20, b: 35, t: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { title: 'Labeled Samples Count', gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            yaxis: { title: 'Accuracy (%)', gridcolor: '#f1f5f9', tickfont: { size: 10, color: '#64748b' } },
            legend: { x: 0.02, y: 0.98, font: { size: 10 } }
        };

        Plotly.react(chartEl, [traceRandom, traceLC, traceLCFD, traceTarget], layout, { responsive: true, displaylogo: false });
    }
}

function renderThresholdsTable(thresholds) {
    const tbody = document.querySelector("#thresholdsTable tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    thresholds.forEach(t => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${t.Dataset}</strong></td>
            <td><span class="badge badge-primary">${t.Target_OA}</span></td>
            <td>${t.Random_Labels}</td>
            <td>${t.LC_Labels}</td>
            <td><strong>${t.LC_FD_Labels}</strong></td>
            <td><span class="badge badge-success">Saved ${t.Saved_Labels} Labels</span></td>
        `;
        tbody.appendChild(tr);
    });
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
let currentSceneViewMode = '3d'; // '3d' or '2d'
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

function renderScene3D() {
    if (!sceneData || !window.Plotly) return;
    const sceneEl = document.getElementById("hsiSpatialScene3D");
    if (!sceneEl) return;

    const [height, width] = sceneData.dimensions;

    // Subsample high-resolution scenes (e.g. Salinas 512x217, Pavia 610x340) for 60fps responsiveness
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

    const traces = [];

    // Base Surface Trace: Real Hyperspectral Spatial Image
    const surfaceTrace = {
        type: 'surface',
        x: xGrid,
        y: yGrid,
        z: zGrid,
        surfacecolor: colorGrid,
        colorscale: sceneData.scene_colorscale,
        cmin: 0,
        cmax: 255,
        showscale: false,
        name: `${currentDataset} Spatial Scene`,
        hoverinfo: 'none'
    };
    traces.push(surfaceTrace);

    // 20 Active Learning Queried Candidates Trace
    const hasSamples = querySelectionData && querySelectionData.selected_samples && querySelectionData.selected_samples.length > 0;
    const badgeEl = document.getElementById("scene3dSamplesBadge");
    if (badgeEl) {
        if (hasSamples) {
            badgeEl.innerHTML = `${querySelectionData.selected_samples.length} Queried Samples: <strong>#1–#${querySelectionData.selected_samples.length}</strong> (Active Learning Rank)`;
        } else {
            badgeEl.innerHTML = `Query Informative Samples to display 20 AL candidate locations in 3D`;
        }
    }

    if (hasSamples && show3DSamples) {
        const samples = querySelectionData.selected_samples;
        const xCoords = samples.map(s => s.coord[1]); // image column = X
        const yCoords = samples.map(s => s.coord[0]); // image row = Y
        const zCoords = new Array(samples.length).fill(4.0); // elevated slightly above plane for 3D visibility
        const textLabels = samples.map(s => `#${s.rank}`);

        const stratName = currentStrategy === 'filter_diversity'
            ? 'LC + Farthest Diversity'
            : currentStrategy === 'least_confidence'
                ? 'Least Confidence'
                : 'Random Sampling';

        // STRICT GROUND TRUTH PRIVACY: Ground truth is NEVER included in Human Annotation Mode
        const customData = samples.map(s => [
            s.rank,
            s.global_index,
            s.coord[1],
            s.coord[0],
            s.least_confidence,
            s.predicted_class,
            stratName
        ]);

        // Selection Sequence Line: Illustrates spatial diversity coverage without implying physical terrain connection
        const lineTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: xCoords,
            y: yCoords,
            z: new Array(samples.length).fill(3.8),
            line: {
                color: 'rgba(245, 158, 11, 0.55)',
                width: 2.5,
                dash: 'dot'
            },
            name: 'Selection Diversity Path',
            hoverinfo: 'none'
        };
        traces.push(lineTrace);

        // Queried Sample Markers (Rank Badges #1–#20)
        const markersTrace = {
            type: 'scatter3d',
            mode: 'markers+text',
            name: 'Queried Samples',
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
                color: '#f59e0b',
                symbol: 'diamond',
                line: { color: '#ffffff', width: 2 }
            },
            customdata: customData,
            hovertemplate:
                '<b>Queried Sample #%{customdata[0]}</b><br>' +
                'Coordinates: (X: %{customdata[2]}, Y: %{customdata[3]})<br>' +
                'Uncertainty (LC): %{customdata[4]:.3f}<br>' +
                'Predicted Class: %{customdata[5]}<br>' +
                'Selection Strategy: %{customdata[6]}<extra></extra>'
        };
        traces.push(markersTrace);
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
                title: { text: 'Visual Depth (Z)', font: { size: 9, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                backgroundcolor: 'rgba(15, 23, 42, 0.5)',
                showbackground: true,
                range: [-0.5, 6]
            },
            camera: {
                eye: { x: 0.0, y: -1.35, z: 1.35 },
                center: { x: 0, y: 0, z: 0 },
                up: { x: 0, y: 0, z: 1 }
            },
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
                    const [rank, gIdx, col, row, lc, predCls, strat] = pt.customdata;
                    updateScene3DInfoCard(rank, col, row, lc, predCls, strat, gIdx);
                    inspectSample(gIdx, rank);
                }
            }
        });

        sceneEl.on('plotly_hover', (evt) => {
            if (evt && evt.points && evt.points.length > 0) {
                const pt = evt.points[0];
                if (pt.customdata) {
                    const [rank, gIdx, col, row, lc, predCls, strat] = pt.customdata;
                    updateScene3DInfoCard(rank, col, row, lc, predCls, strat, gIdx);
                }
            }
        });
    });
}

function updateScene3DInfoCard(rank, x, y, unc, predCls, strat, gIdx) {
    const infoEl = document.getElementById("scene3dSelectedInfo");
    if (!infoEl) return;
    const uncStr = typeof unc === 'number' ? unc.toFixed(3) : unc;
    infoEl.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; width:100%;">
            <div style="font-size:11.5px; line-height:1.5;">
                <span class="badge badge-warning" style="font-weight:700;">Selected Sample #${rank}</span>
                <strong style="margin-left:8px; color:var(--text-primary);">Position:</strong> (x=${x}, y=${y}) &bull;
                <strong style="color:var(--text-secondary);">Uncertainty (LC):</strong> <span style="color:var(--danger); font-weight:700;">${uncStr}</span> &bull;
                <strong style="color:var(--text-secondary);">Predicted class:</strong> <span style="color:var(--primary); font-weight:600;">${predCls}</span> &bull;
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


