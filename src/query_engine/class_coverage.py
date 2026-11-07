import numpy as np
from typing import Dict, Any, Optional

class ClassCoverageController:
    """
    Modular Class-Coverage Constraint for Active Learning in ConformAL-HSI.
    
    Prevents active learning acquisition from starving dominant classes (e.g. Meadows in Pavia)
    or over-querying redundant boundary points of already well-sampled classes.
    
    Operates strictly on candidate-pool pseudo-probabilities and currently labeled counts.
    Never accesses held-out test labels.
    """
    def __init__(self, lambda_cov: float = 0.5, pool_prior_weight: float = 0.7):
        """
        Args:
            lambda_cov: Coverage penalty strength in [0, 1]. 0 = unconstrained.
            pool_prior_weight: Weight given to estimated pool distribution vs uniform prior.
        """
        self.lambda_cov = float(lambda_cov)
        self.pool_prior_weight = float(pool_prior_weight)

    def compute_target_distribution(self, unlabeled_probs: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Estimate true class distribution from candidate pool predictions.
        Blends estimated pool distribution with a mild uniform regularizer.
        """
        pool_est = np.mean(unlabeled_probs, axis=0)
        pool_est = pool_est / np.sum(pool_est)
        uniform_prior = np.ones(num_classes, dtype=np.float64) / num_classes
        target = self.pool_prior_weight * pool_est + (1.0 - self.pool_prior_weight) * uniform_prior
        return target / np.sum(target)

    def apply_coverage_adjustment(
        self,
        base_scores: np.ndarray,
        unlabeled_probs: np.ndarray,
        labeled_labels: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Adjust acquisition scores based on class coverage deficits.
        
        Args:
            base_scores: Unadjusted acquisition scores (e.g. Margin, Conformal, LC) of shape (N_u,).
            unlabeled_probs: Model predicted probabilities for unlabeled pool of shape (N_u, C).
            labeled_labels: Ground-truth labels currently in the labeled pool of shape (N_lab,).
            num_classes: Total number of classes C.
            
        Returns:
            Adjusted scores of shape (N_u,).
        """
        if self.lambda_cov <= 1e-6 or len(labeled_labels) == 0:
            return base_scores

        base_scores = np.asarray(base_scores, dtype=np.float64)
        unlabeled_probs = np.asarray(unlabeled_probs, dtype=np.float64)
        
        # 1. Target distribution
        target_dist = self.compute_target_distribution(unlabeled_probs, num_classes)
        
        # 2. Current labeled empirical distribution (with Laplace smoothing)
        lab_counts = np.bincount(labeled_labels, minlength=num_classes).astype(np.float64)
        total_lab = len(labeled_labels)
        current_dist = (lab_counts + 1.0) / (total_lab + num_classes)
        
        # 3. Class deficit ratio D_c = target_c / current_c
        # D_c > 1 -> class is currently under-represented -> boost query priority
        # D_c < 1 -> class is currently over-represented -> penalize query priority
        deficit_ratios = target_dist / current_dist
        # Clip ratios to avoid extreme weighting
        deficit_ratios = np.clip(deficit_ratios, 0.25, 3.0)
        
        # Dampen by lambda_cov
        class_multipliers = np.power(deficit_ratios, self.lambda_cov)
        
        # 4. Sample-level coverage multiplier: sum_c p(c|x) * multiplier_c
        sample_multipliers = np.dot(unlabeled_probs, class_multipliers)
        
        return base_scores * sample_multipliers

