import numpy as np


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Validate and prepare probability predictions.

    Expected shape:
        (num_samples, num_classes)
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape (num_samples, num_classes)"
        )

    if probabilities.shape[0] == 0:
        raise ValueError("probabilities cannot be empty")

    if probabilities.shape[1] < 2:
        raise ValueError("At least two classes are required")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain NaN or infinite values")

    if np.any(probabilities < 0):
        raise ValueError("probabilities cannot contain negative values")

    # Normalize in case probabilities do not sum exactly to 1.
    row_sums = probabilities.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError("Each sample must contain positive probability mass")

    probabilities = probabilities / row_sums

    return probabilities


def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate normalized predictive entropy.

    Higher score = more uncertain.

    Returns:
        Array with shape (num_samples,)
        Values approximately between 0 and 1.
    """
    probabilities = _validate_probabilities(probabilities)

    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0)

    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1,
    )

    # Normalize entropy so datasets with different numbers
    # of classes remain comparable.
    max_entropy = np.log(probabilities.shape[1])

    return entropy / max_entropy


def least_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Least-confidence uncertainty.

    Higher score = more uncertain.
    """
    probabilities = _validate_probabilities(probabilities)

    confidence = np.max(probabilities, axis=1)

    return 1.0 - confidence


def margin_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    """
    Margin-based uncertainty.

    Difference between the two highest class probabilities.

    A small margin means high uncertainty, so we convert it
    to an uncertainty score:

        uncertainty = 1 - margin

    Higher score = more uncertain.
    """
    probabilities = _validate_probabilities(probabilities)

    sorted_probabilities = np.sort(probabilities, axis=1)

    highest = sorted_probabilities[:, -1]
    second_highest = sorted_probabilities[:, -2]

    margin = highest - second_highest

    return 1.0 - margin


def confidence_thresholding(
    probabilities: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """
    Filter candidate samples based on confidence thresholding.
    
    Returns boolean mask where True indicates sample confidence >= threshold.
    """
    probabilities = _validate_probabilities(probabilities)
    max_probs = np.max(probabilities, axis=1)
    return max_probs >= threshold


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for probability predictions.
    """
    probabilities = _validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64)

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


def get_confidence_statistics(probabilities: np.ndarray) -> dict:
    """
    Compute summary statistics of predictive confidence for logging and monitoring.
    """
    probabilities = _validate_probabilities(probabilities)
    confidences = np.max(probabilities, axis=1)
    return {
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
    }


def compute_uncertainty(
    probabilities: np.ndarray,
    method: str = "entropy",
) -> np.ndarray:
    """
    Unified interface for uncertainty estimation.

    Supported methods:
        entropy
        margin
        least_confidence
    """

    method = method.lower()

    if method == "entropy":
        return entropy_uncertainty(probabilities)

    if method == "margin":
        return margin_uncertainty(probabilities)

    if method in {"least_confidence", "least-confidence", "lc"}:
        return least_confidence(probabilities)

    raise ValueError(
        f"Unknown uncertainty method: {method}. "
        "Choose from: entropy, margin, least_confidence."
    )
