"""
Adaptive Query Controller for SpectralEarth Active Learning Framework.

Dynamically evaluates non-test model state signals (AL round progress,
pool mean confidence, labeled-pool representation gap, and hard-region score)
to determine optimal uncertainty, diversity, and hard-region query weights per round.
"""

from typing import Dict, Tuple, Optional
import numpy as np

from .uncertainty import least_confidence, compute_uncertainty
from .diversity import normalize_embeddings


class AdaptiveQueryController:
    """
    Lightweight, deterministic, and interpretable state-driven Query Controller.

    Uses strictly NON-TEST signals:
        1. AL Round Progress (current_round / total_rounds)
        2. Pool Mean Predictive Confidence
        3. Labeled-Pool Representation Coverage Gap
        4. Persistent Hard-Region Ambiguity Score

    No test labels, ground-truth confusion matrices, or backbone parameter tuning used.
    """

    def __init__(
        self,
        base_start_uncertainty: float = 0.30,
        base_end_uncertainty: float = 0.70,
        enable_hard_region: bool = True,
    ):
        self.base_start_uncertainty = float(base_start_uncertainty)
        self.base_end_uncertainty = float(base_end_uncertainty)
        self.enable_hard_region = enable_hard_region

    def compute_controller_weights(
        self,
        probabilities: np.ndarray,
        current_round: int,
        total_rounds: int,
        embeddings: Optional[np.ndarray] = None,
        labeled_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute normalized query weights (w_uncertainty, w_diversity, w_hard_region)
        based on current round progress and non-test state signals.
        """
        if total_rounds <= 0:
            raise ValueError("total_rounds must be > 0")
        
        progress = current_round / float(max(1, total_rounds - 1)) if total_rounds > 1 else 1.0
        progress = np.clip(progress, 0.0, 1.0)

        # 1. Base scheduled weights
        w_unc_base = self.base_start_uncertainty + progress * (
            self.base_end_uncertainty - self.base_start_uncertainty
        )
        w_div_base = 1.0 - w_unc_base
        w_hard_base = 0.15 if self.enable_hard_region else 0.0

        # 2. Extract non-test pool state signals
        max_probs = np.max(probabilities, axis=1)
        mean_confidence = float(np.mean(max_probs))
        
        # Low confidence signal (< 0.45) indicates model is still globally uncertain;
        # boost diversity to explore unexplored feature sub-spaces.
        confidence_div_boost = 0.0
        if mean_confidence < 0.45:
            confidence_div_boost = 0.10 * (0.45 - mean_confidence) / 0.45

        # Representation gap signal: max distance to existing labeled pool
        rep_gap_boost = 0.0
        if labeled_embeddings is not None and len(labeled_embeddings) > 0 and embeddings is not None:
            norm_cand = normalize_embeddings(embeddings)
            norm_lab = normalize_embeddings(labeled_embeddings)
            sims = np.dot(norm_cand, norm_lab.T)
            sims = np.clip(sims, -1.0, 1.0)
            min_dists = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - sims)))
            max_dist = float(np.max(np.min(min_dists, axis=1)))
            if max_dist > 0.70:
                rep_gap_boost = 0.05

        # 3. Combine and normalize weights
        w_div = w_div_base + confidence_div_boost
        w_unc = w_unc_base
        w_hard = w_hard_base + rep_gap_boost if self.enable_hard_region else 0.0

        total_weight = w_div + w_unc + w_hard
        w_div /= total_weight
        w_unc /= total_weight
        w_hard /= total_weight

        return {
            "w_uncertainty": float(w_unc),
            "w_diversity": float(w_div),
            "w_hard_region": float(w_hard),
            "mean_confidence": float(mean_confidence),
            "progress": float(progress),
        }

    def compute_query_scores(
        self,
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        current_round: int = 0,
        total_rounds: int = 10,
        labeled_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute unified sample selection scores:
            Score(x) = w_unc * U(x) + w_div * D_lab(x) + w_hard * H(x)
        """
        weights = self.compute_controller_weights(
            probabilities=probabilities,
            current_round=current_round,
            total_rounds=total_rounds,
            embeddings=embeddings,
            labeled_embeddings=labeled_embeddings,
        )

        w_unc = weights["w_uncertainty"]
        w_div = weights["w_diversity"]
        w_hard = weights["w_hard_region"]

        # Component 1: Least Confidence Uncertainty U(x)
        unc_raw = least_confidence(probabilities)
        unc_norm = (unc_raw - np.min(unc_raw)) / (np.ptp(unc_raw) + 1e-12)

        # Component 2: Diversity relative to Labeled Pool D_lab(x)
        norm_cand = normalize_embeddings(embeddings)
        if labeled_embeddings is not None and len(labeled_embeddings) > 0:
            norm_lab = normalize_embeddings(labeled_embeddings)
            sims = np.dot(norm_cand, norm_lab.T)
            sims = np.clip(sims, -1.0, 1.0)
            dists_to_lab = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - sims)))
            div_raw = np.min(dists_to_lab, axis=1)
        else:
            centroid = np.mean(norm_cand, axis=0, keepdims=True)
            div_raw = np.linalg.norm(norm_cand - centroid, axis=1)
        
        div_norm = (div_raw - np.min(div_raw)) / (np.ptp(div_raw) + 1e-12)

        # Component 3: Hard-Region Ambiguity Score H(x) = U(x) * D_lab(x)
        hard_raw = unc_norm * div_norm
        hard_norm = (hard_raw - np.min(hard_raw)) / (np.ptp(hard_raw) + 1e-12)

        # Unified Query Score
        final_scores = w_unc * unc_norm + w_div * div_norm + w_hard * hard_norm

        return {
            "final": final_scores,
            "uncertainty": unc_raw,
            "diversity": div_raw,
            "hard_region": hard_raw,
            "weights": weights,
        }

    def select(
        self,
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        k: int,
        current_round: int = 0,
        total_rounds: int = 10,
        labeled_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select K samples using Adaptive Controller state logic.
        Combines top-M candidate filtering with state-weighted query scoring
        and farthest-first diversity traversal.
        """
        scores_dict = self.compute_query_scores(
            embeddings=embeddings,
            probabilities=probabilities,
            current_round=current_round,
            total_rounds=total_rounds,
            labeled_embeddings=labeled_embeddings,
        )

        final_scores = scores_dict["final"]

        # Step 1: Filter top candidate pool M = min(N, max(4k, 50)) using unified query score
        cand_pool_size = min(len(embeddings), max(4 * k, 50))
        top_cand_rel_indices = np.argsort(-final_scores, kind="stable")[:cand_pool_size]
        cand_embeddings = embeddings[top_cand_rel_indices]

        # Step 2: Perform greedy farthest-first diversity selection inside candidate pool
        from .diversity import select_diverse_samples
        selected_in_cand = select_diverse_samples(
            cand_embeddings, k, labeled_embeddings=labeled_embeddings
        )

        selected_indices = top_cand_rel_indices[selected_in_cand]
        return selected_indices.astype(np.int64)
