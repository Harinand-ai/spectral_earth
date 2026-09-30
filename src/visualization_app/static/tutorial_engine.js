/**
 * SPECTRALEARTH INTERACTIVE CONTEXTUAL TUTORIAL ENGINE
 * Professional, game-style contextual guidance built directly into the UI.
 * Highlights real controls, waits for real user actions, explains real results using actual metrics.
 * Non-blocking, unobstructed, viewport-aware, tab-aware, and strictly non-voice.
 */

class TutorialEngine {
    constructor() {
        this.isActive = false;
        this.currentState = "PREPARE";
        this.currentTargetSelector = null;
        this.popoverEl = null;
        this.backdropEl = null;
        this.activeHighlightEl = null;
        this.scrollListenerActive = false;

        // Session tracking for honest metric deltas
        this.sessionMetrics = {
            baselineOa: null,
            baselineF1: null,
            oldOa: null,
            newOa: null,
            targetOa: 70,
            achievedOa: null,
            querySize: 20,
            strategyName: "LC + Farthest Diversity"
        };

        this.initDOM();
    }

    initDOM() {
        // Subtle non-blocking overlay backdrop
        if (!document.getElementById("tutorialBackdrop")) {
            this.backdropEl = document.createElement("div");
            this.backdropEl.id = "tutorialBackdrop";
            this.backdropEl.className = "tutorial-backdrop hidden";
            document.body.appendChild(this.backdropEl);
        } else {
            this.backdropEl = document.getElementById("tutorialBackdrop");
        }

        // Floating contextual popover card
        if (!document.getElementById("tutorialPopover")) {
            this.popoverEl = document.createElement("div");
            this.popoverEl.id = "tutorialPopover";
            this.popoverEl.className = "tutorial-popover hidden";
            document.body.appendChild(this.popoverEl);
        } else {
            this.popoverEl = document.getElementById("tutorialPopover");
        }

        // Intelligent repositioning on resize and scroll
        window.addEventListener("resize", () => {
            if (this.isActive && this.currentTargetSelector) {
                this.positionPopover(this.currentTargetSelector);
            }
        });

        window.addEventListener("scroll", () => {
            if (this.isActive && this.currentTargetSelector) {
                this.positionPopover(this.currentTargetSelector);
            }
        }, { passive: true });
    }

    start() {
        this.isActive = true;
        if (this.popoverEl) this.popoverEl.classList.remove("hidden");

        const btnToggle = document.getElementById("btnToggleGuided");
        if (btnToggle) {
            btnToggle.innerText = "Close Guide";
            btnToggle.className = "btn btn-secondary btn-sm";
        }

        // Determine current tab and state intelligently
        const activeTab = document.querySelector(".nav-tab.active");
        const tabId = activeTab ? activeTab.getAttribute("data-tab") : "al-pipeline-tab";

        if (tabId === "cross-sensor-tab") {
            this.currentState = "CROSS_SENSOR";
        } else if (tabId === "experiments-tab") {
            this.currentState = "EXPERIMENTS";
        } else if (tabId === "report-tab") {
            this.currentState = "REPORT";
        } else {
            // Check Active Learning state
            const roundEl = document.getElementById("metricRound");
            const roundText = roundEl ? roundEl.innerText : "0 / 5";
            const roundNum = parseInt(roundText.split("/")[0].trim()) || 0;

            const hasQueryRows = document.querySelectorAll("#querySelectionTable tbody tr:not(.empty-placeholder)").length > 0;
            const isPredictDone = (typeof hasPredicted !== "undefined" && hasPredicted) ||
                (document.getElementById("sceneActiveContent") && document.getElementById("sceneActiveContent").style.display === "block");

            if (roundNum > 0) {
                this.currentState = "RETRAIN_RESULT";
            } else if (hasQueryRows) {
                this.currentState = "INSPECT_SAMPLE";
            } else if (isPredictDone) {
                this.currentState = "QUERY";
            } else {
                this.currentState = "PREPARE";
            }
        }

        this.renderState();
    }

    stop() {
        this.isActive = false;
        this.removeScrollListener();
        if (this.backdropEl) this.backdropEl.classList.add("hidden");
        if (this.popoverEl) this.popoverEl.classList.add("hidden");
        if (this.activeHighlightEl) {
            this.activeHighlightEl.classList.remove("tutorial-target-highlight");
            this.activeHighlightEl = null;
        }
        this.currentTargetSelector = null;

        const btnToggle = document.getElementById("btnToggleGuided");
        if (btnToggle) {
            btnToggle.innerText = "Guided Tutorial";
            btnToggle.className = "btn btn-primary btn-sm";
        }
    }

    renderState() {
        if (!this.isActive) return;

        // Clean up scroll listeners if moving between states
        if (this.currentState !== "SCROLL_TO_SAMPLES") {
            this.removeScrollListener();
        }

        switch (this.currentState) {
            case "PREPARE":
                this.renderPrepareStage();
                break;
            case "PREDICT":
                this.renderPredictStage();
                break;
            case "PREDICT_RESULT":
                this.renderPredictResultStage();
                break;
            case "QUERY":
                this.renderQueryStage();
                break;
            case "QUERY_RESULT":
                this.renderQueryResultStage();
                break;
            case "SCROLL_TO_SAMPLES":
                this.renderScrollToSamplesStage();
                break;
            case "INSPECT_SAMPLE":
                this.renderInspectSampleStage();
                break;
            case "INSPECT_MODAL":
                this.renderInspectModalStage();
                break;
            case "ANNOTATE":
                this.renderAnnotateStage();
                break;
            case "ANNOTATE_RESULT":
                this.renderAnnotateResultStage();
                break;
            case "RETRAIN":
                this.renderRetrainStage();
                break;
            case "RETRAIN_RESULT":
                this.renderRetrainResultStage();
                break;
            case "LABEL_EFFICIENCY":
                this.renderLabelEfficiencyStage();
                break;
            case "TARGET_OA":
                this.renderTargetOaStage();
                break;
            case "CROSS_SENSOR":
                this.renderCrossSensorStage();
                break;
            case "CROSS_SENSOR_RESULT":
                this.renderCrossSensorResultStage();
                break;
            case "EXPERIMENTS":
                this.renderExperimentsStage();
                break;
            case "REPORT":
                this.renderReportStage();
                break;
            case "COMPLETE":
                this.renderCompleteStage();
                break;
            default:
                this.renderPrepareStage();
        }
    }

    // =========================================================================
    // STEP 1 — PREPARE THE EXPERIMENT (Grouped Configuration)
    // =========================================================================
    renderPrepareStage() {
        const dsSel = document.getElementById("datasetSelect");
        const stratSel = document.getElementById("strategySelect");
        const modeSel = document.getElementById("modeToggleSelect");
        const kSel = document.getElementById("queryBatchSelect");

        const dsVal = dsSel ? dsSel.options[dsSel.selectedIndex]?.text.split("(")[0].trim() : "Indian Pines";
        const stratVal = stratSel ? stratSel.options[stratSel.selectedIndex]?.text.split("(")[0].trim() : "LC + Farthest Diversity";
        const modeVal = modeSel ? (modeSel.value === "human" ? "Human Annotation" : "Simulated Oracle") : "Simulated Oracle";
        const kVal = kSel ? kSel.value : "20";

        this.sessionMetrics.querySize = parseInt(kVal) || 20;
        this.sessionMetrics.strategyName = stratVal;

        this.currentTargetSelector = "#expControlBar";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Experiment Guide</span>
                    <h4 class="tutorial-title">Prepare the Experiment</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Choose the dataset, annotation mode, Active Learning strategy and query size.</p>
                    <div class="tutorial-checklist">
                        <div class="check-item"><span class="check-icon">✓</span> Dataset: <strong>${dsVal}</strong></div>
                        <div class="check-item"><span class="check-icon">✓</span> Mode: <strong>${modeVal}</strong></div>
                        <div class="check-item"><span class="check-icon">✓</span> Strategy: <strong>${stratVal}</strong></div>
                        <div class="check-item"><span class="check-icon">✓</span> Query Size: <strong>${kVal} samples</strong></div>
                    </div>
                    <div class="tutorial-instruction-box">
                        Your experiment is configured. Generate the initial prediction.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.stop()">Close</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('PREDICT')">Next: Prediction →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 2 — INITIAL PREDICTION
    // =========================================================================
    renderPredictStage() {
        this.currentTargetSelector = "#btnInitialPredict";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Baseline Evaluation</span>
                    <h4 class="tutorial-title">Initial Prediction</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Generate the initial prediction. This gives us the classifier’s starting performance before we acquire additional labels.</p>
                    <div class="tutorial-instruction-box">
                        👉 Click the highlighted <strong>⚡ Initial Prediction</strong> button above.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('PREPARE')">Back</button>
                    <span style="font-size:11px; color:var(--text-muted);">Waiting for click...</span>
                </div>
            </div>
        `;
    }

    renderPredictResultStage() {
        this.currentTargetSelector = "#cardHsiScene";
        this.positionPopover(this.currentTargetSelector);

        const oaVal = document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "52.40%";
        const f1Val = document.getElementById("metricF1") ? document.getElementById("metricF1").innerText : "0.4520";

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Prediction Ready</span>
                    <h4 class="tutorial-title">Initial Prediction Result</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">The initial prediction is ready.</p>
                    <p class="tutorial-text">The map shows the classifier’s current prediction across the scene before additional labelled samples are acquired.</p>
                    <p class="tutorial-text">SpectralEarth provides the pretrained hyperspectral feature representation. Its backbone remains frozen while the lightweight classifier is adapted to the dataset.</p>
                    <div class="tutorial-metrics-badge">
                        Baseline Accuracy: <strong>OA = ${oaVal}</strong> &bull; <strong>Macro F1 = ${f1Val}</strong>
                    </div>
                    <div class="tutorial-instruction-box">
                        👉 Now run Query Sampling.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('PREDICT')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('QUERY')">Next: Query Sampling →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 3 — ACTIVE LEARNING QUERY
    // =========================================================================
    renderQueryStage() {
        const kSel = document.getElementById("queryBatchSelect");
        const kVal = kSel ? kSel.value : (this.sessionMetrics.querySize || 20);
        this.sessionMetrics.querySize = parseInt(kVal) || 20;

        this.currentTargetSelector = "#btnRunQuery";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Active Learning</span>
                    <h4 class="tutorial-title">Query Sampling</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Now select the Active Learning strategy and query size, then run Query Sampling.</p>
                    <div class="tutorial-instruction-box">
                        👉 Click the highlighted <strong>1. Query Samples (k=${kVal})</strong> button above.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('PREDICT_RESULT')">Back</button>
                    <span style="font-size:11px; color:var(--text-muted);">Waiting for query...</span>
                </div>
            </div>
        `;
    }

    renderQueryResultStage() {
        const k = this.sessionMetrics.querySize || 20;

        // Check if sample table is currently below viewport
        const sampleCard = document.getElementById("cardQuerySelection");
        const isBelow = sampleCard ? this.isElementBelowViewport(sampleCard) : false;

        if (isBelow) {
            this.currentState = "SCROLL_TO_SAMPLES";
            this.renderScrollToSamplesStage();
            return;
        }

        this.currentTargetSelector = "#cardHsiScene";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#eff6ff; color:#2563eb;">Candidates Selected</span>
                    <h4 class="tutorial-title">Query Result</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text"><strong>${k} samples were selected.</strong></p>
                    <p class="tutorial-text">These are candidate samples for annotation because the selected Active Learning strategy considers them informative.</p>
                    <p class="tutorial-text">Least Confidence focuses on uncertain predictions, while the diversity component helps reduce redundant selections.</p>
                    <div class="tutorial-instruction-box">
                        👉 Choose one of the selected samples to inspect.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('QUERY')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('INSPECT_SAMPLE')">Go to Samples ↓</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // EXPLICIT SCROLLING TO SELECTED SAMPLES
    // =========================================================================
    renderScrollToSamplesStage() {
        const k = this.sessionMetrics.querySize || 20;

        this.currentTargetSelector = "#cardHsiScene";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#eff6ff; color:#2563eb;">Candidates Selected</span>
                    <h4 class="tutorial-title">Query Result</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text"><strong>${k} samples were selected.</strong></p>
                    <p class="tutorial-text">These samples are candidates for annotation because the selected Active Learning strategy considers them informative.</p>
                    <p class="tutorial-text">Least Confidence focuses on uncertain predictions, while the diversity component helps reduce redundant selections.</p>
                    <div class="tutorial-scroll-alert">
                        👇 <strong>Scroll down</strong> to the selected samples.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('QUERY')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('INSPECT_SAMPLE')">I've Scrolled ↓</button>
                </div>
            </div>
        `;

        // Listen for user scroll until samples table becomes visible
        this.attachScrollListener();
    }

    attachScrollListener() {
        this.removeScrollListener();
        this.scrollListenerActive = true;

        this.onScrollHandler = () => {
            const sampleCard = document.getElementById("cardQuerySelection");
            if (sampleCard && !this.isElementBelowViewport(sampleCard)) {
                this.removeScrollListener();
                this.currentState = "INSPECT_SAMPLE";
                this.renderInspectSampleStage();
            }
        };

        window.addEventListener("scroll", this.onScrollHandler, { passive: true });
    }

    removeScrollListener() {
        if (this.scrollListenerActive && this.onScrollHandler) {
            window.removeEventListener("scroll", this.onScrollHandler);
            this.scrollListenerActive = false;
        }
    }

    isElementBelowViewport(el) {
        const rect = el.getBoundingClientRect();
        return rect.top > (window.innerHeight - 100);
    }

    // =========================================================================
    // STEP 4 — INSPECT A SAMPLE
    // =========================================================================
    renderInspectSampleStage() {
        this.currentTargetSelector = "#querySelectionTable";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Sample Inspection</span>
                    <h4 class="tutorial-title">Choose a Sample</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Choose one of the selected samples to inspect its spatial patch, spectral signature, prediction and class probabilities.</p>
                    <div class="tutorial-instruction-box">
                        👉 Click <strong>Inspect</strong> on any sample row in the table below.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('QUERY')">Back</button>
                    <span style="font-size:11px; color:var(--text-muted);">Waiting for Inspect click...</span>
                </div>
            </div>
        `;
    }

    // Modal Opened: WAIT for user. DO NOT jump to retrain or cover the modal!
    renderInspectModalStage() {
        this.currentTargetSelector = "#sampleModal .modal-box";
        this.positionPopover(this.currentTargetSelector, "modal-dock");

        this.popoverEl.innerHTML = `
            <div class="tutorial-card" style="width:290px;">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Inspecting Sample</span>
                    <h4 class="tutorial-title">Sample Details</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">This sample contains a 15×15 spatial patch, a spectral reflectance signature and class probabilities.</p>
                    <p class="tutorial-text">The spectral curve shows how the pixel responds across wavelengths, while the probabilities show the classifier’s confidence across classes.</p>
                    <div class="tutorial-instruction-box">
                        👉 Provide a label for this sample (or click Save Label / close modal).
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('INSPECT_SAMPLE')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('ANNOTATE')">Next: Label →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 5 — ANNOTATION
    // =========================================================================
    renderAnnotateStage() {
        const isModalOpen = document.getElementById("sampleModal")?.classList.contains("active");
        this.currentTargetSelector = isModalOpen ? "#modalAnnotationSection" : "#thAnnotationCol";
        this.positionPopover(this.currentTargetSelector, isModalOpen ? "modal-dock" : "normal");

        const modeSel = document.getElementById("modeToggleSelect");
        const isHuman = modeSel && modeSel.value === "human";

        const modeNote = isHuman
            ? "Human Annotation mode: ground truth remains hidden to ensure unassisted manual evaluation."
            : "Simulated Oracle mode uses the available ground-truth label to simulate the annotation process.";

        this.popoverEl.innerHTML = `
            <div class="tutorial-card" style="width:290px;">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Annotation</span>
                    <h4 class="tutorial-title">Provide Label</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Now provide a label for this sample. This represents the annotation stage of the Active Learning workflow.</p>
                    <p class="tutorial-text" style="font-size:11px; color:var(--text-secondary); background:var(--bg-subtle); padding:5px 7px; border-radius:4px;">${modeNote}</p>
                    <div class="tutorial-instruction-box">
                        👉 ${isModalOpen ? "Choose a class and click <strong>Save Label</strong>, or close the modal." : "Assign class annotations in the table."}
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('INSPECT_SAMPLE')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('ANNOTATE_RESULT')">Label Submitted →</button>
                </div>
            </div>
        `;
    }

    renderAnnotateResultStage() {
        // If modal is still open, close it cleanly
        const modal = document.getElementById("sampleModal");
        if (modal && modal.classList.contains("active")) {
            modal.classList.remove("active");
        }

        this.currentTargetSelector = "#btnSubmitAnnotate";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Label Pool Updated</span>
                    <h4 class="tutorial-title">Sample Labelled</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">The sample has been added to the labelled pool.</p>
                    <p class="tutorial-text">This lets us improve the classifier without manually labelling the entire scene.</p>
                    <div class="tutorial-instruction-box">
                        👉 Retrain the classifier using the updated labelled pool.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('ANNOTATE')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('RETRAIN')">Proceed to Retrain →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 6 — RETRAIN
    // =========================================================================
    renderRetrainStage() {
        this.currentTargetSelector = "#btnSubmitAnnotate";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Model Update</span>
                    <h4 class="tutorial-title">Retrain Classifier</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Now retrain the lightweight classifier using the updated labelled pool.</p>
                    <div class="tutorial-instruction-box">
                        👉 Click the highlighted <strong>2. Retrain Classifier</strong> button above.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('ANNOTATE_RESULT')">Back</button>
                    <span style="font-size:11px; color:var(--text-muted);">Waiting for retraining...</span>
                </div>
            </div>
        `;
    }

    renderRetrainResultStage(payload = null) {
        this.currentTargetSelector = "#metricSummaryBox";
        this.positionPopover(this.currentTargetSelector);

        const oldOa = payload?.old_oa || this.sessionMetrics.oldOa || "52.40";
        const newOa = payload?.new_oa || this.sessionMetrics.newOa || (document.getElementById("metricOA") ? document.getElementById("metricOA").innerText.replace("%", "") : "65.93");

        const oldNum = parseFloat(oldOa) || 52.4;
        const newNum = parseFloat(newOa) || 65.9;

        let deltaComment = "";
        if (newNum < oldNum) {
            deltaComment = `Overall Accuracy decreased slightly in this round (from ${oldOa}% to ${newOa}%). Active Learning does not guarantee that every individual query will improve accuracy, so we evaluate performance across multiple rounds and datasets.`;
        } else {
            deltaComment = `Overall Accuracy changed from ${oldOa}% to ${newOa}%. The SpectralEarth backbone remained frozen; the lightweight classifier was updated using the newly labelled samples.`;
        }

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Retraining Complete</span>
                    <h4 class="tutorial-title">Retraining Result</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Retraining is complete.</p>
                    <p class="tutorial-text">${deltaComment}</p>
                    <div class="tutorial-instruction-box">
                        👉 Evaluate label efficiency on the accuracy progression chart.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('RETRAIN')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('LABEL_EFFICIENCY')">Label Efficiency ↓</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 7 — LABEL EFFICIENCY
    // =========================================================================
    renderLabelEfficiencyStage() {
        this.currentTargetSelector = "#cardProgression";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Progression Analysis</span>
                    <h4 class="tutorial-title">Label Efficiency</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Scroll down to Label Efficiency.</p>
                    <p class="tutorial-text">This shows how classification performance changes as we increase the number of labelled samples.</p>
                    <p class="tutorial-text">The curve plots test accuracy across active learning cycles, demonstrating label efficiency on the frozen representation.</p>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('RETRAIN_RESULT')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('TARGET_OA')">Target Accuracy →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 8 — TARGET ACCURACY
    // =========================================================================
    renderTargetOaStage() {
        const targetInput = document.getElementById("targetOaInput");
        const targetVal = targetInput ? targetInput.value : "70";
        const oaVal = document.getElementById("metricOA") ? document.getElementById("metricOA").innerText : "52.40%";
        const badge = document.getElementById("targetStatusBadge");
        const statusText = (badge && badge.innerText.includes("NOT")) ? "Not Reached" : "Reached ✓";

        this.currentTargetSelector = "#targetOaContainer";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Benchmark Goal</span>
                    <h4 class="tutorial-title">Target Accuracy</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Set the target Overall Accuracy you want to reach, then evaluate whether the current label budget is enough.</p>
                    <div class="tutorial-metrics-badge">
                        The target was <strong>${targetVal}%</strong>.<br>
                        The current result is <strong>${oaVal}</strong>.<br>
                        The target is <strong>${statusText}</strong>.
                    </div>
                    <div class="tutorial-instruction-box" style="margin-top:8px;">
                        👉 Switch tabs above to test <strong>Cross-Sensor Adaptation</strong> or examine <strong>Experimental Evidence</strong>.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('LABEL_EFFICIENCY')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.switchToTab('tabCrossSensor')">Cross-Sensor Tab →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 9 — CROSS-SENSOR ADAPTATION
    // =========================================================================
    renderCrossSensorStage() {
        this.currentTargetSelector = "#transferPairSelect";
        this.positionPopover(this.currentTargetSelector);

        const pairSelect = document.getElementById("transferPairSelect");
        const pairName = pairSelect ? pairSelect.value : "Indian Pines -> Pavia University";

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Domain Transfer</span>
                    <h4 class="tutorial-title">Cross-Sensor Adaptation</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Now we’ll demonstrate cross-sensor adaptation.</p>
                    <p class="tutorial-text">Choose the source and target datasets, then run the adaptation.</p>
                    <div class="tutorial-instruction-box">
                        👉 Selected: <strong>${pairName}</strong>. Review 1D spectral alignment and few-shot adaptation results below.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.switchToTab('tabActiveLearning')">Active Learning</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('CROSS_SENSOR_RESULT')">Explain Result →</button>
                </div>
            </div>
        `;
    }

    renderCrossSensorResultStage() {
        this.currentTargetSelector = "#resamplerChart";
        this.positionPopover(this.currentTargetSelector);

        const csFewShot = document.getElementById("csFewShot") ? document.getElementById("csFewShot").innerText : "49.84%";

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Spectral Alignment</span>
                    <h4 class="tutorial-title">Adaptation Analysis</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Different sensors can use different spectral band configurations.</p>
                    <p class="tutorial-text">The system first aligns the spectral representation using 1D band resampling, then adapts the lightweight classifier using target data.</p>
                    <div class="tutorial-metrics-badge">
                        Few-Shot Target Adaptation: <strong>${csFewShot} OA</strong>
                    </div>
                    <div class="tutorial-instruction-box">
                        👉 Click the <strong>Experimental Evidence</strong> tab above to view rigorous benchmark tables.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('CROSS_SENSOR')">Back</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.switchToTab('tabExperiments')">Experiments Tab →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 10 — EXPERIMENTS (Sub-View Aware)
    // =========================================================================
    renderExperimentsStage() {
        const expTypeSel = document.getElementById("expTypeSelect");
        const expType = expTypeSel ? expTypeSel.value : "benchmark";

        let expComment = "";
        if (expType === "benchmark") {
            expComment = "Benchmark results compare the framework across the evaluated datasets and Active Learning baselines.";
        } else if (expType === "ablation") {
            expComment = "Ablation studies show which components actually help. More components do not automatically produce better performance.";
        } else if (expType === "calibration") {
            expComment = "Calibration evaluates whether the model’s confidence matches the reliability of its predictions.";
        } else if (expType === "cost") {
            expComment = "This section shows the computational cost of the frozen backbone and lightweight classifier.";
        } else {
            expComment = "This section displays empirical evidence across 5-seed multi-trial experiments.";
        }

        this.currentTargetSelector = "#expTypeSelect";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag">Evidence</span>
                    <h4 class="tutorial-title">Experimental Evidence</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">Now we can examine the experimental evidence behind the framework.</p>
                    <div class="tutorial-callout-box" style="background:#f8fafc; border-left:3px solid var(--primary); padding:6px 8px; margin:6px 0; font-size:11px;">
                        ${expComment}
                    </div>
                    <div class="tutorial-instruction-box">
                        👉 Switch between Benchmark, Ablation, Calibration, and Cost using the dropdown.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.switchToTab('tabCrossSensor')">Cross-Sensor</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.switchToTab('tabReport')">Research Report →</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // STEP 11 — RESEARCH REPORT
    // =========================================================================
    renderReportStage() {
        this.currentTargetSelector = "#reportIframe";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#eff6ff; color:#2563eb;">Audit Document</span>
                    <h4 class="tutorial-title">Research Report</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text">The Research Report contains the detailed methodology, experimental evidence, limitations and reproducibility information.</p>
                    <div class="tutorial-instruction-box">
                        You can read the report directly in this frame or open it in a new window.
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.advanceTo('COMPLETE')">Next: Complete →</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.advanceTo('COMPLETE')">Finish Walkthrough</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // TUTORIAL COMPLETION (Rule 19)
    // =========================================================================
    renderCompleteStage() {
        this.currentTargetSelector = "#expControlBar";
        this.positionPopover(this.currentTargetSelector);

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Walkthrough Complete</span>
                    <h4 class="tutorial-title">Experiment Complete</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    <p class="tutorial-text"><strong>Experiment walkthrough complete.</strong></p>
                    <p class="tutorial-text">You have completed the main workflow:</p>
                    <div style="background:var(--bg-subtle); padding:6px 8px; border-radius:4px; font-size:11px; margin-bottom:8px; line-height:1.6;">
                        Prediction → Active Learning → Annotation → Retraining → Evaluation → Cross-Sensor Adaptation
                    </div>
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.switchToTab('tabActiveLearning'); window.tutorialEngine.advanceTo('PREPARE');">Restart Guide</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.stop()">Close Guide</button>
                </div>
            </div>
        `;
    }

    // =========================================================================
    // RESULT COMMENT & EVENT NOTIFIER (Crash-Proof API)
    // =========================================================================
    showResultComment(title, htmlContent, nextInstruction = "", targetSelector = null) {
        if (!this.isActive) return;

        if (targetSelector) {
            this.currentTargetSelector = targetSelector;
            this.positionPopover(targetSelector);
        }

        this.popoverEl.innerHTML = `
            <div class="tutorial-card">
                <div class="tutorial-header">
                    <span class="tutorial-step-tag" style="background:#ecfdf5; color:#059669;">Result</span>
                    <h4 class="tutorial-title">${title}</h4>
                    <button class="tutorial-close-btn" onclick="window.tutorialEngine.stop()">&times;</button>
                </div>
                <div class="tutorial-body">
                    ${htmlContent}
                    ${nextInstruction ? `<div class="tutorial-instruction-box">👉 ${nextInstruction}</div>` : ""}
                </div>
                <div class="tutorial-footer">
                    <button class="btn btn-secondary btn-sm" onclick="window.tutorialEngine.stop()">Dismiss</button>
                    <button class="btn btn-primary btn-sm" onclick="window.tutorialEngine.handleNextStep()">Continue →</button>
                </div>
            </div>
        `;
    }

    notifyAction(actionType, payload = null) {
        if (!this.isActive) return;

        switch (actionType) {
            case "INITIAL_PREDICTION_DONE":
                if (payload) {
                    this.sessionMetrics.baselineOa = payload.oa;
                    this.sessionMetrics.baselineF1 = payload.f1;
                    this.sessionMetrics.oldOa = payload.oa;
                }
                this.currentState = "PREDICT_RESULT";
                this.renderState();
                break;

            case "QUERY_SAMPLING_DONE":
                if (payload && payload.sampleCount) {
                    this.sessionMetrics.querySize = payload.sampleCount;
                }
                this.currentState = "QUERY_RESULT";
                this.renderState();
                break;

            case "SAMPLE_INSPECTED":
                // Open inspection modal: show explanation and WAIT (DO NOT auto-advance!)
                this.currentState = "INSPECT_MODAL";
                this.renderState();
                break;

            case "SAMPLE_ANNOTATED":
                // Meaningful action performed: transition to annotated result
                this.currentState = "ANNOTATE_RESULT";
                this.renderState();
                break;

            case "RETRAIN_DONE":
                if (payload) {
                    this.sessionMetrics.oldOa = payload.old_oa || this.sessionMetrics.oldOa;
                    this.sessionMetrics.newOa = payload.new_oa || this.sessionMetrics.newOa;
                }
                this.currentState = "RETRAIN_RESULT";
                this.renderRetrainResultStage(payload);
                break;

            case "TARGET_SET":
                if (payload) {
                    this.sessionMetrics.targetOa = payload.target_oa;
                    this.sessionMetrics.achievedOa = payload.achieved_oa;
                }
                this.currentState = "TARGET_OA";
                this.renderState();
                break;

            case "TAB_CHANGED":
                if (payload && payload.tabId) {
                    if (payload.tabId === "cross-sensor-tab") {
                        this.currentState = "CROSS_SENSOR";
                    } else if (payload.tabId === "experiments-tab") {
                        this.currentState = "EXPERIMENTS";
                    } else if (payload.tabId === "report-tab") {
                        this.currentState = "REPORT";
                    } else if (payload.tabId === "al-pipeline-tab") {
                        // Resume appropriate AL state without confusion
                        const roundEl = document.getElementById("metricRound");
                        const roundNum = parseInt(roundEl?.innerText?.split("/")[0]?.trim()) || 0;
                        if (roundNum > 0) this.currentState = "RETRAIN_RESULT";
                        else if (typeof hasQueried !== "undefined" && hasQueried) this.currentState = "INSPECT_SAMPLE";
                        else if (typeof hasPredicted !== "undefined" && hasPredicted) this.currentState = "QUERY";
                        else this.currentState = "PREPARE";
                    }
                    this.renderState();
                }
                break;

            case "EXPERIMENT_TYPE_CHANGED":
                if (this.currentState === "EXPERIMENTS") {
                    this.renderExperimentsStage();
                }
                break;

            case "CROSS_SENSOR_DONE":
                if (this.currentState === "CROSS_SENSOR") {
                    this.currentState = "CROSS_SENSOR_RESULT";
                    this.renderState();
                }
                break;
        }
    }

    advanceTo(stateName) {
        this.currentState = stateName;
        this.renderState();
    }

    handleNextStep() {
        const progression = {
            "PREPARE": "PREDICT",
            "PREDICT": "PREDICT_RESULT",
            "PREDICT_RESULT": "QUERY",
            "QUERY": "QUERY_RESULT",
            "QUERY_RESULT": "INSPECT_SAMPLE",
            "SCROLL_TO_SAMPLES": "INSPECT_SAMPLE",
            "INSPECT_SAMPLE": "INSPECT_MODAL",
            "INSPECT_MODAL": "ANNOTATE",
            "ANNOTATE": "ANNOTATE_RESULT",
            "ANNOTATE_RESULT": "RETRAIN",
            "RETRAIN": "RETRAIN_RESULT",
            "RETRAIN_RESULT": "LABEL_EFFICIENCY",
            "LABEL_EFFICIENCY": "TARGET_OA",
            "TARGET_OA": "CROSS_SENSOR",
            "CROSS_SENSOR": "CROSS_SENSOR_RESULT",
            "CROSS_SENSOR_RESULT": "EXPERIMENTS",
            "EXPERIMENTS": "REPORT",
            "REPORT": "COMPLETE",
            "COMPLETE": "PREPARE"
        };
        const next = progression[this.currentState] || "PREPARE";
        this.advanceTo(next);
    }

    switchToTab(tabId) {
        const tabEl = document.getElementById(tabId);
        if (tabEl) tabEl.click();
    }

    // =========================================================================
    // INTELLIGENT NON-OVERLAPPING POPOVER POSITIONING (Rule 6 & 16)
    // =========================================================================
    positionPopover(selector, mode = "normal") {
        let targetEl = document.querySelector(selector);
        if (!targetEl) targetEl = document.getElementById("expControlBar") || document.body;

        // Reset previous highlight
        if (this.activeHighlightEl) {
            this.activeHighlightEl.classList.remove("tutorial-target-highlight");
        }

        if (!targetEl || targetEl === document.body) {
            this.popoverEl.style.position = "fixed";
            this.popoverEl.style.top = "50%";
            this.popoverEl.style.left = "50%";
            this.popoverEl.style.transform = "translate(-50%, -50%)";
            return;
        }

        // Apply gentle highlight
        targetEl.classList.add("tutorial-target-highlight");
        this.activeHighlightEl = targetEl;

        const popoverWidth = 310;
        const targetRect = targetEl.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;

        // Special handling for modals: dock to side without covering close button or charts
        if (mode === "modal-dock") {
            const modalBox = document.querySelector("#sampleModal .modal-box");
            if (modalBox) {
                const boxRect = modalBox.getBoundingClientRect();
                let left = boxRect.right + 12;
                if (left + popoverWidth > viewportWidth - 16) {
                    left = Math.max(16, boxRect.left - popoverWidth - 12);
                }
                let top = Math.max(16, boxRect.top);
                this.popoverEl.style.position = "fixed";
                this.popoverEl.style.top = `${top}px`;
                this.popoverEl.style.left = `${left}px`;
                this.popoverEl.style.width = `${popoverWidth}px`;
                this.popoverEl.style.transform = "none";
                return;
            }
        }

        // Determine placement: prefer below, then above, then right/left
        let top = targetRect.bottom + 8;
        let left = targetRect.left;

        // Prevent horizontal clipping
        if (left + popoverWidth > viewportWidth - 16) {
            left = viewportWidth - popoverWidth - 16;
        }
        if (left < 16) left = 16;

        // If placing below exceeds screen, place above target
        if (top + 180 > viewportHeight) {
            top = targetRect.top - 190;
            if (top < 16) {
                // If above also clips, place to right or left side
                top = Math.max(16, targetRect.top);
                left = targetRect.right + 12;
                if (left + popoverWidth > viewportWidth - 16) {
                    left = Math.max(16, targetRect.left - popoverWidth - 12);
                }
            }
        }

        this.popoverEl.style.position = "fixed";
        this.popoverEl.style.top = `${Math.max(16, top)}px`;
        this.popoverEl.style.left = `${Math.max(16, left)}px`;
        this.popoverEl.style.width = `${popoverWidth}px`;
        this.popoverEl.style.transform = "none";
    }
}

window.tutorialEngine = new TutorialEngine();
