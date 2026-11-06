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
        self.seen_pseudo_indices = set()
        self.round_history = []

    def reset_history(self):
        """Reset cumulative bookkeeping for a new experiment or seed."""
        self.seen_pseudo_indices = set()
        self.round_history = []

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
                - "per_class_counts": Dict[int, int] of accepted counts per class (post-filtering)
                - "total_accepted": int (round pseudo count)
                - "round_pseudo_count": int (accepted in this round)
                - "cumulative_pseudo_count": int (unique samples pseudo-labeled across rounds)
                - "mean_confidence": float
        """
        probabilities = np.asarray(probabilities, dtype=np.float32)
        unlabeled_indices = np.asarray(unlabeled_indices, dtype=np.int64)

        empty_per_class = {c: 0 for c in range(num_classes)}

        if len(probabilities) == 0 or len(unlabeled_indices) == 0:
            return {
                "pseudo_indices": np.empty((0,), dtype=np.int64),
                "pseudo_labels": np.empty((0,), dtype=np.int64),
                "pseudo_confidences": np.empty((0,), dtype=np.float32),
                "per_class_counts": empty_per_class,
                "total_accepted": 0,
                "round_pseudo_count": 0,
                "cumulative_pseudo_count": len(self.seen_pseudo_indices),
                "mean_confidence": 0.0,
            }

        preds = np.argmax(probabilities, axis=1)
        confs = np.max(probabilities, axis=1)

        accepted_rel_indices = []

        # Step 1: Apply threshold and per-class cap
        for c in range(num_classes):
            mask = (preds == c) & (confs >= self.confidence_threshold)
            rel_indices = np.where(mask)[0]
            if len(rel_indices) > 0:
                # Sort descending by confidence
                sorted_rel = rel_indices[np.argsort(-confs[rel_indices])]
                take = sorted_rel[:self.max_pseudo_per_class]
                accepted_rel_indices.extend(take.tolist())

        if len(accepted_rel_indices) == 0:
            return {
                "pseudo_indices": np.empty((0,), dtype=np.int64),
                "pseudo_labels": np.empty((0,), dtype=np.int64),
                "pseudo_confidences": np.empty((0,), dtype=np.float32),
                "per_class_counts": empty_per_class,
                "total_accepted": 0,
                "round_pseudo_count": 0,
                "cumulative_pseudo_count": len(self.seen_pseudo_indices),
                "mean_confidence": 0.0,
            }

        accepted_rel_indices = np.array(accepted_rel_indices, dtype=np.int64)
        pseudo_global_indices = unlabeled_indices[accepted_rel_indices]
        pseudo_labels = preds[accepted_rel_indices]
        pseudo_confidences = confs[accepted_rel_indices]

        # Step 2: Guard against duplicate indices or accidental inclusion of genuine labels
        if labeled_indices is not None and len(labeled_indices) > 0:
            genuine_set = set(np.asarray(labeled_indices).tolist())
            clean_mask = [g not in genuine_set for g in pseudo_global_indices]
            if not all(clean_mask):
                clean_mask = np.array(clean_mask, dtype=bool)
                pseudo_global_indices = pseudo_global_indices[clean_mask]
                pseudo_labels = pseudo_labels[clean_mask]
                pseudo_confidences = pseudo_confidences[clean_mask]

        # Step 3: Ensure internal uniqueness (no duplicate global indices)
        if len(pseudo_global_indices) > 0:
            _, unique_pos = np.unique(pseudo_global_indices, return_index=True)
            if len(unique_pos) < len(pseudo_global_indices):
                unique_pos = np.sort(unique_pos)
                pseudo_global_indices = pseudo_global_indices[unique_pos]
                pseudo_labels = pseudo_labels[unique_pos]
                pseudo_confidences = pseudo_confidences[unique_pos]

        # Step 4: Recompute per_class_counts accurately from the final accepted pseudo_labels!
        per_class_counts = {c: int(np.sum(pseudo_labels == c)) for c in range(num_classes)}

        # Step 5: Update cumulative tracking (without double-counting identical samples)
        self.seen_pseudo_indices.update(pseudo_global_indices.tolist())

        record = {
            "pseudo_indices": pseudo_global_indices,
            "pseudo_labels": pseudo_labels,
            "pseudo_confidences": pseudo_confidences,
            "per_class_counts": per_class_counts,
            "total_accepted": len(pseudo_global_indices),
            "round_pseudo_count": len(pseudo_global_indices),
            "cumulative_pseudo_count": len(self.seen_pseudo_indices),
            "mean_confidence": float(np.mean(pseudo_confidences)) if len(pseudo_confidences) > 0 else 0.0,
        }
        self.round_history.append(record)
        return record
