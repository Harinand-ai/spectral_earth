from typing import Tuple
import numpy as np


def get_adaptive_weights(
    current_round: int,
    total_rounds: int,
    start_uncertainty: float = 0.30,
    end_uncertainty: float = 0.80,
) -> Tuple[float, float]:
    """
    Compute adaptive uncertainty/diversity weights.

    Early rounds:
        More importance to diversity.

    Later rounds:
        More importance to uncertainty.

    Returns
    -------
    uncertainty_weight, diversity_weight
    """

    if total_rounds <= 0:
        raise ValueError("total_rounds must be greater than 0")

    current_round = max(0, min(current_round, total_rounds - 1))

    if not 0.0 <= start_uncertainty <= 1.0:
        raise ValueError(
            "start_uncertainty must be between 0 and 1"
        )

    if not 0.0 <= end_uncertainty <= 1.0:
        raise ValueError(
            "end_uncertainty must be between 0 and 1"
        )

    if total_rounds == 1:
        progress = 1.0
    else:
        progress = current_round / (total_rounds - 1)

    uncertainty_weight = (
        start_uncertainty
        + progress * (end_uncertainty - start_uncertainty)
    )

    diversity_weight = 1.0 - uncertainty_weight

    return float(uncertainty_weight), float(diversity_weight)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores into [0, 1].
    """

    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array")

    if len(scores) == 0:
        raise ValueError("scores cannot be empty")

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinite values")

    minimum = np.min(scores)
    maximum = np.max(scores)

    if np.isclose(maximum, minimum):
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def combine_scores(
    uncertainty_scores: np.ndarray,
    diversity_scores: np.ndarray,
    current_round: int,
    total_rounds: int,
    start_uncertainty: float = 0.30,
    end_uncertainty: float = 0.80,
) -> np.ndarray:
    """
    Combine uncertainty and diversity scores using adaptive weights.

    Higher final score = higher query priority.
    """

    uncertainty_scores = np.asarray(
        uncertainty_scores,
        dtype=np.float64,
    )

    diversity_scores = np.asarray(
        diversity_scores,
        dtype=np.float64,
    )

    if uncertainty_scores.shape != diversity_scores.shape:
        raise ValueError(
            "uncertainty_scores and diversity_scores "
            "must have identical shapes"
        )

    uncertainty_scores = normalize_scores(
        uncertainty_scores
    )

    diversity_scores = normalize_scores(
        diversity_scores
    )

    uncertainty_weight, diversity_weight = (
        get_adaptive_weights(
            current_round=current_round,
            total_rounds=total_rounds,
            start_uncertainty=start_uncertainty,
            end_uncertainty=end_uncertainty,
        )
    )

    combined = (
        uncertainty_weight * uncertainty_scores
        + diversity_weight * diversity_scores
    )

    return combined