import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

def compute_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification and calibration metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    aa = float(np.mean(per_class_r))
    macro_p = float(np.mean(per_class_p))
    macro_r = float(np.mean(per_class_r))
    macro_f1 = float(np.mean(per_class_f1))

    # Calibration metrics
    confidences = np.max(y_prob, axis=1)
    mean_conf = float(np.mean(confidences))
    ece = compute_ece(y_prob, y_true)

    per_class_metrics = {}
    for i in range(num_classes):
        name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        per_class_metrics[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    return {
        "OverallAccuracy": float(oa),
        "AverageAccuracy": float(aa),
        "Precision": float(macro_p),
        "Recall": float(macro_r),
        "F1Score": float(macro_f1),
        "MacroF1": float(macro_f1),
        "Macro_F1": float(macro_f1),
        "macro_f1": float(macro_f1),
        "Kappa": float(kappa),
        "ECE": float(ece),
        "MeanConfidence": mean_conf,
        "ConfusionMatrix": cm.tolist(),
        "PerClass": per_class_metrics,
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / max(1, n_samples))

    return float(ece)


def compute_aulc(labeled_counts: List[int], metric_values: List[float]) -> float:
    """
    Calculate Normalized Area Under Learning Curve (AULC).
    
    AULC = trapezoidal area divided by total budget range (max_labeled - min_labeled).
    Value ranges between 0.0 and 1.0 (or 0 and 100 if metric is percentage).
    """
    if len(labeled_counts) < 2:
        return float(metric_values[0]) if len(metric_values) > 0 else 0.0

    counts = np.array(labeled_counts, dtype=np.float64)
    values = np.array(metric_values, dtype=np.float64)

    if hasattr(np, "trapezoid"):
        total_area = np.trapezoid(values, counts)
    elif hasattr(np, "trapz"):
        total_area = np.trapz(values, counts)
    else:
        from scipy.integrate import trapezoid
        total_area = trapezoid(values, counts)
    total_range = counts[-1] - counts[0]
    if total_range <= 0:
        return float(values[-1])

    return float(total_area / total_range)
