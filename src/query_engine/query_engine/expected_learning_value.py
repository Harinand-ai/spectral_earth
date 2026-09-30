"""
Expected Learning Value / Value of Information Query Strategy.
Computes lightweight proxies for expected classifier improvement:
1. Predictive Uncertainty U(x)
2. Representation Novelty N(x) (distance to labeled pool)
3. Expected Classifier Gradient Signal G(x) (analytical expected loss gradient w.r.t classifier head)
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .uncertainty import least_confidence


def compute_expected_gradient_signal(
    probabilities: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute analytical expected gradient magnitude w.r.t 2-layer MLP classifier head.
    
    For cross-entropy loss, the expected gradient magnitude of the final layer linear weights
    under the current probability distribution P(y|x) is given by:
        G(x) = (1 - sum_c p_c^2) * ||h(x)||_2
    where (1 - sum_c p_c^2) is Gini impurity / predictive variance, and ||h(x)||_2 is
    the norm of the hidden representation (or embedding as proxy).
    
    Parameters:
        probabilities: (N, C) predicted class probabilities.
        embeddings: (N, D) feature embeddings (optional scale factor).
        
    Returns:
        (N,) expected gradient score.
    """
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    # Gini impurity = 1 - sum(p_c^2)
    gini_impurity = 1.0 - np.sum(probabilities ** 2, axis=1)
    
    if embeddings is not None and embeddings.shape[0] == probabilities.shape[0]:
        feat_norm = np.linalg.norm(embeddings, axis=1)
        # Normalize feature norm to avoid extreme scale imbalance
        feat_norm = feat_norm / (np.max(feat_norm) + 1e-8)
        grad_signal = gini_impurity * feat_norm
    else:
        grad_signal = gini_impurity
        
    return grad_signal


def compute_representation_novelty(
    unlabeled_embeddings: np.ndarray,
    labeled_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute novelty of unlabeled embeddings as minimum Euclidean distance to current labeled pool.
    
    Parameters:
        unlabeled_embeddings: (M, D) candidate embeddings.
        labeled_embeddings: (N_l, D) labeled pool embeddings.
        
    Returns:
        (M,) minimum distance to labeled pool.
    """
    if labeled_embeddings is None or len(labeled_embeddings) == 0:
        # Fallback to distance from candidate mean
        centroid = np.mean(unlabeled_embeddings, axis=0, keepdims=True)
        dists = np.linalg.norm(unlabeled_embeddings - centroid, axis=1)
        return dists

    # Efficient vectorized distance computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a_sq = np.sum(unlabeled_embeddings ** 2, axis=1, keepdims=True)  # (M, 1)
    b_sq = np.sum(labeled_embeddings ** 2, axis=1, keepdims=True).T # (1, N_l)
    ab = np.dot(unlabeled_embeddings, labeled_embeddings.T)          # (M, N_l)
    
    sq_dists = np.maximum(0.0, a_sq + b_sq - 2 * ab)
    min_dists = np.sqrt(np.min(sq_dists, axis=1))
    return min_dists


class ExpectedLearningValueController:
    """
    Controller for Expected Learning Value / Information Gain selection.
    
    Selects top-k samples maximizing:
        V(x) = w_U * norm(U(x)) + w_N * norm(N(x)) + w_G * norm(G(x))
    
    Supports candidate filtering: evaluates N(x) and G(x) only on top candidate_multiplier * k
    uncertain candidates to keep query time ultra-low (< 0.1s).
    """

    def __init__(
        self,
        w_uncertainty: float = 0.40,
        w_novelty: float = 0.40,
        w_gradient: float = 0.20,
        candidate_multiplier: int = 4,
    ):
        tot = w_uncertainty + w_novelty + w_gradient
        self.w_u = w_uncertainty / tot
        self.w_n = w_novelty / tot
        self.w_g = w_gradient / tot
        self.candidate_multiplier = max(1, candidate_multiplier)

    def select_samples(
        self,
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        k: int,
        labeled_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select top-k samples using Expected Learning Value.
        
        Returns:
            Tuple of (selected_unlabeled_indices [k], detailed_info_dict)
        """
        n_samples = len(embeddings)
        if n_samples <= k:
            return np.arange(n_samples), {"method": "expected_learning_value", "filtered": False}

        # Step 1: Inexpensive uncertainty filtering
        u_scores = least_confidence(probabilities)
        m_candidates = min(n_samples, self.candidate_multiplier * k)
        
        if m_candidates < n_samples:
            cand_indices = np.argsort(-u_scores)[:m_candidates]
            cand_embeddings = embeddings[cand_indices]
            cand_probs = probabilities[cand_indices]
            cand_u = u_scores[cand_indices]
        else:
            cand_indices = np.arange(n_samples)
            cand_embeddings = embeddings
            cand_probs = probabilities
            cand_u = u_scores

        # Step 2: Compute Representation Novelty on candidates
        cand_n = compute_representation_novelty(cand_embeddings, labeled_embeddings)

        # Step 3: Compute Expected Gradient Signal on candidates
        cand_g = compute_expected_gradient_signal(cand_probs, cand_embeddings)

        # Step 4: Min-Max Normalize components across candidate pool
        def min_max(arr):
            rng = np.max(arr) - np.min(arr)
            return (arr - np.min(arr)) / (rng + 1e-8)

        norm_u = min_max(cand_u)
        norm_n = min_max(cand_n)
        norm_g = min_max(cand_g)

        # Step 5: Compute Unified Expected Learning Value Score
        v_scores = (self.w_u * norm_u) + (self.w_n * norm_n) + (self.w_g * norm_g)

        # Step 6: Select Top-K from candidates
        top_k_rel = np.argsort(-v_scores)[:k]
        selected_unlabeled_indices = cand_indices[top_k_rel]

        details = {
            "method": "expected_learning_value",
            "candidate_count": m_candidates,
            "weights": {"w_u": self.w_u, "w_n": self.w_n, "w_g": self.w_g},
            "candidate_indices": cand_indices,
            "v_scores": v_scores,
            "norm_u": norm_u,
            "norm_n": norm_n,
            "norm_g": norm_g,
        }

        return selected_unlabeled_indices, details
