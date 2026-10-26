from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch


class ConfidencePseudoLabeler:
    """
    Confidence-Based Pseudo-Labeler for Active Learning in Hyperspectral Classification.
    
    Scores unlabeled pool candidates using predicted probabilities, accepts samples
    exceeding confidence threshold tau, and enforces a per-class cap to prevent
    domination by majority classes.

    Mathematical formulation:
        \\hat{y}(x) = \\arg\\max_c P(y=c \\mid x)
        q(x) = \\max_c P(y=c \\mid x)
        Candidate accepted if q(x) >= tau
        Per-class cap: at most K_cap samples per predicted class c, sorted by q(x) descending.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        max_pseudo_per_class: int = 15,
        pseudo_weight: float = 0.2,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.max_pseudo_per_class = int(max_pseudo_per_class)
        self.pseudo_weight = float(pseudo_weight)

    def select_pseudo_labels(
        self,
        probabilities: np.ndarray,
        unlabeled_indices: np.ndarray,
        num_classes: int,
        labeled_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Select high-confidence pseudo-labels from the unlabeled pool.

        Args:
            probabilities: [N_u, C] predicted class probabilities for unlabeled samples.
            unlabeled_indices: [N_u] global dataset indices corresponding to the rows of probabilities.
            num_classes: Total number of classes.
            labeled_indices: Optional array of global indices already in the genuine labeled pool
                             (used for assertion/safeguard to prevent duplicates).

        Returns:
            Dict containing:
                - "pseudo_indices": np.ndarray of accepted global indices
                - "pseudo_labels": np.ndarray of integer class predictions
                - "pseudo_confidences": np.ndarray of confidence values q(x)
                - "per_class_counts": Dict[int, int] of accepted counts per class
                - "total_accepted": int
                - "mean_confidence": float
        """
        probabilities = np.asarray(probabilities, dtype=np.float32)
        unlabeled_indices = np.asarray(unlabeled_indices, dtype=np.int64)

        if len(probabilities) == 0 or len(unlabeled_indices) == 0:
            return {
                "pseudo_indices": np.empty((0,), dtype=np.int64),
                "pseudo_labels": np.empty((0,), dtype=np.int64),
                "pseudo_confidences": np.empty((0,), dtype=np.float32),
                "per_class_counts": {c: 0 for c in range(num_classes)},
                "total_accepted": 0,
                "mean_confidence": 0.0,
            }

        preds = np.argmax(probabilities, axis=1)
        confs = np.max(probabilities, axis=1)

        accepted_rel_indices = []
        per_class_counts = {c: 0 for c in range(num_classes)}

        # Apply threshold and per-class cap
        for c in range(num_classes):
            mask = (preds == c) & (confs >= self.confidence_threshold)
            rel_indices = np.where(mask)[0]
            if len(rel_indices) > 0:
                # Sort descending by confidence
                sorted_rel = rel_indices[np.argsort(-confs[rel_indices])]
                take = sorted_rel[:self.max_pseudo_per_class]
                accepted_rel_indices.extend(take.tolist())
                per_class_counts[c] = len(take)

        if len(accepted_rel_indices) == 0:
            return {
                "pseudo_indices": np.empty((0,), dtype=np.int64),
                "pseudo_labels": np.empty((0,), dtype=np.int64),
                "pseudo_confidences": np.empty((0,), dtype=np.float32),
                "per_class_counts": per_class_counts,
                "total_accepted": 0,
                "mean_confidence": 0.0,
            }

        accepted_rel_indices = np.array(accepted_rel_indices, dtype=np.int64)
        pseudo_global_indices = unlabeled_indices[accepted_rel_indices]
        pseudo_labels = preds[accepted_rel_indices]
        pseudo_confidences = confs[accepted_rel_indices]

        # Guard against duplicate indices or accidental inclusion of genuine labels
        if labeled_indices is not None and len(labeled_indices) > 0:
            genuine_set = set(labeled_indices.tolist())
            clean_mask = [g not in genuine_set for g in pseudo_global_indices]
            if not all(clean_mask):
                pseudo_global_indices = pseudo_global_indices[clean_mask]
                pseudo_labels = pseudo_labels[clean_mask]
                pseudo_confidences = pseudo_confidences[clean_mask]

        return {
            "pseudo_indices": pseudo_global_indices,
            "pseudo_labels": pseudo_labels,
            "pseudo_confidences": pseudo_confidences,
            "per_class_counts": per_class_counts,
            "total_accepted": len(pseudo_global_indices),
            "mean_confidence": float(np.mean(pseudo_confidences)) if len(pseudo_confidences) > 0 else 0.0,
        }
