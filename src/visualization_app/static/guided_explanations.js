/**
 * GUIDED DEMONSTRATION EXPLANATION LIBRARY
 * Centralized templates for event-driven guided voice assistant.
 * Uses simple English while preserving core scientific terminology.
 * All numeric values are dynamically injected from real backend/frontend results.
 */

const GUIDED_EXPLANATIONS = {
    welcome: {
        title: "Welcome to SpectralEarth Guided Demonstration",
        speech: "Welcome. We’ll demonstrate how our framework uses a frozen SpectralEarth foundation model, Active Learning, and a lightweight classifier to learn from a limited number of labelled samples. First, prepare the experiment. Choose your dataset, choose the annotation mode, select the Active Learning strategy, set the query size, and then start the initial prediction. You can configure these settings while the dataset loads.",
        status: "Experiment Preparation — Select dataset, annotation mode, strategy & query size."
    },

    dataset_loading: {
        title: "Dataset Loading...",
        speech_template: "{dataset} is selected and loading in the background. While it loads, you can choose your annotation mode, Active Learning strategy, or target accuracy.",
        status_template: "Loading {dataset}... You can configure settings while loading."
    },

    dataset_loaded: {
        title: "Dataset Ready & Initial Prediction",
        speech_template: "{dataset} is ready. SpectralEarth provides a frozen 2048-dimensional representation, and our lightweight classifier makes the initial classification. Now scroll down to the Active Learning section to configure and run sample querying.",
        status_template: "{dataset} loaded ({dimensions_str}, {bands} bands, {classes} classes). Scroll down to Active Learning."
    },

    query_completed: {
        title: "Informative Samples Queried",
        speech_template: "{k} samples have been selected using {strategy}. The red markers show the candidate samples chosen for annotation. Our Least Confidence plus Feature Diversity strategy considers both model uncertainty and diversity between selected samples. Now scroll down to the selected samples table and inspect one of them.",
        status_template: "{k} samples queried using {strategy}. Scroll down to inspect a sample."
    },

    sample_inspected: {
        title: "Sample Inspected",
        speech_template: "This sample shows three useful pieces of information. The spatial patch provides context, the 200-band spectral curve shows reflectance across wavelengths, and class probabilities show model uncertainty. Now provide the label for this sample in the table and click Retrain Classifier.",
        status_template: "Inspecting rank #{rank} ({coord_r}, {coord_c}). Provide label and click Retrain Classifier."
    },

    retraining_completed: {
        title: "Lightweight Classifier Retrained",
        speech_template: "The classifier head has been retrained. Before retraining, Overall Accuracy was {previous_oa} percent. After retraining, it is {new_oa} percent. That is a {improvement} percentage point change using {labels_added} additional labels, while the SpectralEarth backbone remained frozen.",
        status_template: "Retrained MLP head ({labels_added} labels added). Overall Accuracy: {previous_oa}% → {new_oa}%."
    },

    target_reached: {
        title: "Target Accuracy Reached!",
        speech_template: "The target Overall Accuracy of {target_oa} percent has been reached. The system required {num_labeled} total labeled samples to reach this target.",
        status_template: "Target OA of {target_oa}% reached with {num_labeled} labeled samples."
    },

    target_not_reached: {
        title: "Target Accuracy Not Reached Yet",
        speech_template: "The current Overall Accuracy is {new_oa} percent, while the target is {target_oa} percent. Scroll down and click Query Sampling to query another batch of informative samples.",
        status_template: "Current OA: {new_oa}%, Target: {target_oa}%. Click Query Sampling for next round."
    },

    cross_sensor_tab: {
        title: "Cross-Sensor Adaptation Mode",
        speech_template: "Scroll down to Cross-Sensor Adaptation. Choose the source and target datasets, review the spectral alignment settings, and run the adaptation.",
        status_template: "Cross-Sensor view active. Choose source and target sensors."
    },

    cross_sensor_selected: {
        title: "Cross-Sensor Pair Selected",
        speech_template: "The two sensors have different spectral configurations. {source_dataset} has {source_bands} bands, while {target_dataset} has {target_bands} bands. We align the spectral bands using 1D spectral linear interpolation and adapt the lightweight classifier using target data.",
        status_template: "Resampling {source_dataset} ({source_bands} bands) → {target_dataset} ({target_bands} bands)."
    },

    cross_sensor_adapted: {
        title: "Few-Shot Target Adaptation Results",
        speech_template: "The lightweight classifier has been adapted using target-domain labeled samples. Overall Accuracy reached {few_shot_oa} percent on {target_dataset} using {few_shot_labels} target domain labels, while the SpectralEarth backbone remained frozen.",
        status_template: "Adapted to {target_dataset}: {few_shot_oa}% OA with {few_shot_labels} labels."
    },

    experiments_tab: {
        title: "Experimental Evidence View",
        speech_template: "You are viewing the Experimental Evidence section. Here you can inspect 5-seed benchmark results, label efficiency curves, ablation studies, calibration profiles, and computational latency.",
        status_template: "Viewing 5-seed research benchmark artifacts."
    },

    report_tab: {
        title: "Research Report View",
        speech_template: "This page displays the automatically generated 12-section scientific report summarizing live session metrics and experimental findings.",
        status_template: "Viewing 12-section research report."
    },

    explain_this_al: {
        title: "Active Learning Overview",
        speech_template: "You are currently viewing the Active Learning module for {dataset}. The model is currently at Round {round} with {num_labeled} labeled samples, achieving an Overall Accuracy of {oa} percent.",
        status_template: "Active Learning for {dataset} (Round {round}, {num_labeled} labeled samples, OA: {oa}%)."
    },

    wrong_action: {
        title: "Action Pending",
        speech_template: "{message}",
        status_template: "{message}"
    },

    operation_failed: {
        title: "Operation Failed",
        speech_template: "The operation could not be completed. Please try again.",
        status_template: "Operation failed. Please try again."
    }
};

if (typeof module !== "undefined" && module.exports) {
    module.exports = GUIDED_EXPLANATIONS;
}
