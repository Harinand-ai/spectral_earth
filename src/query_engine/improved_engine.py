"""
ConformAL-HSI: Improved Modular Query Engine.

Implements the Proposed Integrated Active Learning Architecture:
1. Conformal Credibility Gated Margin Uncertainty
2. Adaptive Class-Coverage Deficit Multiplier (No Test Leakage)
3. Vectorized Hyperspherical Cosine Pool Diversity
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from src.conformal.split_conformal import SplitConformalPredictor
from src.query_engine.class_coverage import ClassCoverageController


class ImprovedConformalALEngine:
    """
    Proposed Integrated ConformAL-HSI Acquisition Engine.
    
    Synthesizes:
    - Two-class Margin Uncertainty (high boundary sharpness)
    - Conformal Credibility Weighting (outlier/noise rejection + ambiguity boosting)
    - Dynamic Class-Coverage Modulation (prevents dominant/minority starvation)
    - Vectorized Hyperspherical Cosine Pool Diversity (non-redundant hypersphere coverage)
    """
    def __init__(
        self,
        alpha: float = 0.10,
        lambda_cov: float = 0.5,
        pool_prior_weight: float = 0.7,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conformal_predictor = SplitConformalPredictor(alpha=alpha, score_type="softmax_error")
        self.coverage_controller = ClassCoverageController(lambda_cov=lambda_cov, pool_prior_weight=pool_prior_weight)

    def calibrate_conformal(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """Calibrate conformal quantile threshold on validation pool."""
        return self.conformal_predictor.calibrate(val_probs, val_labels)

    def select_samples(
        self,
        strategy_code: str,
        k: int,
        unlabeled_idx: np.ndarray,
        train_feats_l2: torch.Tensor,
        train_lbls: torch.Tensor,
        labeled_idx: np.ndarray,
        model: torch.nn.Module,
        num_classes: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, float]:
        """
        Main sample selection entrypoint.
        
        Args:
            strategy_code: One of ['random', 'lc_fd', 'margin', 'margin_cosine', 
                                   'conformal_unc', 'conformal_cosine', 'proposed_conformal_al']
            k: Number of samples to query.
            unlabeled_idx: Global dataset indices of unlabeled candidate pool.
            train_feats_l2: Pre-normalized L2 feature embeddings of shape (N, 2048).
            train_lbls: Training ground truth labels of shape (N,).
            labeled_idx: Global dataset indices of currently labeled pool.
            model: Trained classifier head.
            num_classes: Number of classes C.
            rng: RandomState instance.
            
        Returns:
            Tuple of (selected_global_indices, acquisition_runtime_seconds).
        """
        import time
        if len(unlabeled_idx) <= k:
            return unlabeled_idx, 0.0

        t0 = time.time()

        # Strategy 1: Random Sampling Baseline
        if strategy_code == "random":
            chosen = rng.choice(len(unlabeled_idx), size=k, replace=False)
            return unlabeled_idx[chosen], time.time() - t0

        # Step 1: Forward pass to extract predicted class probabilities on unlabeled pool
        model.eval()
        u_probs_list = []
        with torch.no_grad():
            for bi in range(0, len(unlabeled_idx), 2048):
                bx = train_feats_l2[unlabeled_idx[bi : bi + 2048]].to(self.device)
                logits = model(bx)
                u_probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(u_probs_list, axis=0)

        # Strategy 2: Original LC + FD Baseline
        if strategy_code == "lc_fd":
            max_p = np.max(probs, axis=1)
            unc = 1.0 - max_p
            cand_size = min(len(unlabeled_idx), max(4 * k, 50))
            top_cand = np.argsort(-unc, kind="stable")[:cand_size]
            cand_raw = train_feats_l2[unlabeled_idx[top_cand]].to(self.device)

            with torch.no_grad():
                sel_cand = [0]
                dists = torch.norm(cand_raw - cand_raw[0:1], dim=1)
                for _ in range(1, k):
                    farthest = torch.argmax(dists).item()
                    sel_cand.append(farthest)
                    new_dists = torch.norm(cand_raw - cand_raw[farthest : farthest + 1], dim=1)
                    dists = torch.minimum(dists, new_dists)

            return unlabeled_idx[top_cand[sel_cand]], time.time() - t0

        # Strategy 3: Pure Margin Sampling
        sorted_probs = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])

        if strategy_code == "margin":
            top_cand = np.argsort(-margin_unc, kind="stable")[:k]
            return unlabeled_idx[top_cand], time.time() - t0

        # Strategy 4: Margin + Cosine Diversity
        if strategy_code == "margin_cosine":
            cand_size = min(len(unlabeled_idx), max(4 * k, 200))
            top_cand = np.argsort(-margin_unc, kind="stable")[:cand_size]
            selected_sub = self._vectorized_cosine_diversity(
                cand_idx=top_cand,
                unlabeled_idx=unlabeled_idx,
                labeled_idx=labeled_idx,
                train_feats_l2=train_feats_l2,
                k=k
            )
            return unlabeled_idx[top_cand[selected_sub]], time.time() - t0

        # Strategy 5: Conformal Uncertainty Only
        conf_unc = self.conformal_predictor.compute_conformal_uncertainty(probs)
        if strategy_code == "conformal_unc":
            top_cand = np.argsort(-conf_unc, kind="stable")[:k]
            return unlabeled_idx[top_cand], time.time() - t0

        # Strategy 6: Conformal + Cosine Diversity
        if strategy_code == "conformal_cosine":
            cand_size = min(len(unlabeled_idx), max(4 * k, 200))
            top_cand = np.argsort(-conf_unc, kind="stable")[:cand_size]
            selected_sub = self._vectorized_cosine_diversity(
                cand_idx=top_cand,
                unlabeled_idx=unlabeled_idx,
                labeled_idx=labeled_idx,
                train_feats_l2=train_feats_l2,
                k=k
            )
            return unlabeled_idx[top_cand[selected_sub]], time.time() - t0

        # Strategy 7: Proposed Integrated ConformAL-HSI Framework
        # (Conformal Credibility Gated Margin + Class Coverage + Cosine Diversity)
        if strategy_code == "proposed_conformal_al":
            pred_sets = self.conformal_predictor.predict_sets(probs)
            set_sizes = np.array([len(s) for s in pred_sets], dtype=np.float64)

            # Credibility weights:
            # - Empty sets (outliers / severe noise): downweight to 0.20
            # - Exact two-class ambiguity (ideal margin): boost to 1.25
            # - Multi-class ambiguity (>= 3 classes): weight 1.10
            # - Confident singleton (= 1 class): weight 0.85
            credibility_weights = np.ones(len(probs), dtype=np.float64)
            credibility_weights[set_sizes == 0] = 0.20
            credibility_weights[set_sizes == 1] = 0.85
            credibility_weights[set_sizes == 2] = 1.25
            credibility_weights[set_sizes >= 3] = 1.10

            gated_margin_score = margin_unc * credibility_weights

            # Apply class-coverage modulation to avoid dominant class starvation
            labeled_labels = train_lbls[labeled_idx].numpy()
            adjusted_score = self.coverage_controller.apply_coverage_adjustment(
                base_scores=gated_margin_score,
                unlabeled_probs=probs,
                labeled_labels=labeled_labels,
                num_classes=num_classes
            )

            # Filter top candidate pool
            cand_size = min(len(unlabeled_idx), max(4 * k, 200))
            top_cand = np.argsort(-adjusted_score, kind="stable")[:cand_size]

            # Vectorized Cosine Farthest-First Traversal
            selected_sub = self._vectorized_cosine_diversity(
                cand_idx=top_cand,
                unlabeled_idx=unlabeled_idx,
                labeled_idx=labeled_idx,
                train_feats_l2=train_feats_l2,
                k=k
            )
            return unlabeled_idx[top_cand[selected_sub]], time.time() - t0

        raise ValueError(f"Unknown strategy code: {strategy_code}")

    def _vectorized_cosine_diversity(
        self,
        cand_idx: np.ndarray,
        unlabeled_idx: np.ndarray,
        labeled_idx: np.ndarray,
        train_feats_l2: torch.Tensor,
        k: int
    ) -> List[int]:
        """
        Vectorized Hyperspherical Cosine Diversity Traversal.
        
        Iteratively selects candidates maximizing distance to both labeled pool and accumulating batch.
        Runs fully on GPU in < 2 milliseconds.
        """
        cand_l2 = train_feats_l2[unlabeled_idx[cand_idx]].to(self.device)
        lab_l2 = train_feats_l2[labeled_idx].to(self.device)

        with torch.no_grad():
            # Initial distance to labeled pool: 1.0 - max_{l in L} cos(x, l)
            dots_to_lab = torch.matmul(cand_l2, lab_l2.t())
            max_dot, _ = torch.max(dots_to_lab, dim=1)
            dists = 1.0 - max_dot

            # First selection is the candidate farthest from labeled pool
            sel_cand = [torch.argmax(dists).item()]
            dists[sel_cand[-1]] = -1.0

            # Iteratively update minimum distance to newly selected samples
            for _ in range(1, k):
                new_dots = torch.matmul(cand_l2, cand_l2[sel_cand[-1] : sel_cand[-1] + 1].t()).squeeze(1)
                dists = torch.minimum(dists, 1.0 - new_dots)
                dists[sel_cand[-1]] = -1.0
                sel_cand.append(torch.argmax(dists).item())

        return sel_cand

