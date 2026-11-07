import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

class SplitConformalPredictor:
    """
    Split Conformal Prediction Engine for ConformAL-HSI.
    
    Provides mathematically rigorous distribution-free uncertainty calibration
    and prediction set generation with finite-sample coverage guarantees.
    
    Under exchangeability:
        P(Y_test in C(X_test)) >= 1 - alpha
        
    Attributes:
        alpha (float): Significance level (e.g. 0.10 for 90% target coverage).
        nonconformity_quantile (float): Conformal nonconformity threshold computed on calibration set.
        is_calibrated (bool): Whether the predictor has been calibrated on validation data.
    """
    def __init__(self, alpha: float = 0.10, score_type: str = "softmax_error"):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.target_coverage = 1.0 - self.alpha
        self.score_type = score_type.lower()
        self.q_hat: Optional[float] = None
        self.is_calibrated: bool = False
        self.calibration_size: int = 0
        self.cal_scores: Optional[np.ndarray] = None

    def compute_nonconformity_scores(self, probs: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute nonconformity scores.
        
        Args:
            probs: Predicted class probabilities of shape (N, C).
            labels: Ground-truth class labels of shape (N,) if available.
            
        Returns:
            If labels is provided: 1D array of shape (N,) with nonconformity scores for true labels.
            If labels is None: 2D array of shape (N, C) with nonconformity scores for all classes.
        """
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        if self.score_type == "softmax_error":
            # s(x, y) = 1 - p(y|x)
            if labels is not None:
                labels = np.asarray(labels, dtype=np.int64)
                N = len(labels)
                true_probs = probs[np.arange(N), labels]
                return 1.0 - true_probs
            else:
                return 1.0 - probs
        elif self.score_type == "margin":
            # s(x, y) = max_{c != y} p(c|x) - p(y|x)
            if labels is not None:
                labels = np.asarray(labels, dtype=np.int64)
                N = len(labels)
                scores = np.zeros(N, dtype=np.float64)
                for i in range(N):
                    p_true = probs[i, labels[i]]
                    p_other = np.delete(probs[i], labels[i])
                    scores[i] = np.max(p_other) - p_true
                return scores
            else:
                # All classes
                N, C = probs.shape
                scores = np.zeros((N, C), dtype=np.float64)
                for c in range(C):
                    p_c = probs[:, c]
                    mask = np.ones(C, dtype=bool)
                    mask[c] = False
                    p_other = np.max(probs[:, mask], axis=1)
                    scores[:, c] = p_other - p_c
                return scores
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Fit conformal threshold on held-out validation/calibration pool.
        
        Uses standard finite-sample correction:
            q_level = ceil((N_cal + 1) * (1 - alpha)) / N_cal
        """
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int64)
        n_cal = len(val_labels)
        if n_cal == 0:
            raise ValueError("Calibration set cannot be empty.")
            
        self.calibration_size = n_cal
        self.cal_scores = self.compute_nonconformity_scores(val_probs, val_labels)
        
        # Finite sample quantile calculation
        q_level = np.ceil((n_cal + 1) * self.target_coverage) / n_cal
        q_level = min(1.0, max(0.0, q_level))
        
        # Use higher quantile to guarantee conservative finite-sample coverage
        self.q_hat = float(np.quantile(self.cal_scores, q_level, method="higher"))
        self.is_calibrated = True
        return self.q_hat

    def predict_sets(self, probs: np.ndarray) -> List[List[int]]:
        """
        Construct prediction sets C(x) for given class probabilities.
        
        C(x) = {c : s(x, c) <= q_hat}
        """
        if not self.is_calibrated or self.q_hat is None:
            raise RuntimeError("SplitConformalPredictor must be calibrated before generating prediction sets.")
            
        all_scores = self.compute_nonconformity_scores(probs, labels=None)
        in_set = (all_scores <= self.q_hat)
        
        prediction_sets = []
        for i in range(len(probs)):
            classes_in_set = np.where(in_set[i])[0].tolist()
            # If set is empty (very uncertain/OOD), include argmax class as standard fallback
            if len(classes_in_set) == 0:
                classes_in_set = [int(np.argmax(probs[i]))]
            prediction_sets.append(classes_in_set)
            
        return prediction_sets

    def evaluate_coverage(self, test_probs: np.ndarray, test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate empirical coverage and prediction set metrics on held-out test data.
        """
        test_labels = np.asarray(test_labels, dtype=np.int64)
        pred_sets = self.predict_sets(test_probs)
        n_test = len(test_labels)
        
        covered = [test_labels[i] in pred_sets[i] for i in range(n_test)]
        empirical_coverage = float(np.mean(covered) * 100.0)
        
        set_sizes = np.array([len(s) for s in pred_sets], dtype=np.float64)
        mean_set_size = float(np.mean(set_sizes))
        median_set_size = float(np.median(set_sizes))
        singleton_pct = float(np.mean(set_sizes == 1) * 100.0)
        multi_pct = float(np.mean(set_sizes > 1) * 100.0)
        
        # Class-conditional coverage
        unique_classes = np.unique(test_labels)
        class_coverage = {}
        for c in unique_classes:
            c_mask = (test_labels == c)
            if c_mask.sum() > 0:
                class_coverage[int(c)] = float(np.mean([covered[i] for i in np.where(c_mask)[0]]) * 100.0)
                
        return {
            "empirical_coverage_pct": empirical_coverage,
            "target_coverage_pct": self.target_coverage * 100.0,
            "mean_set_size": mean_set_size,
            "median_set_size": median_set_size,
            "singleton_pct": singleton_pct,
            "multi_pct": multi_pct,
            "q_hat": self.q_hat,
            "calibration_size": self.calibration_size,
            "class_coverage": class_coverage
        }

    def compute_conformal_uncertainty(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute Conformal Active Learning Acquisition Scores.
        
        Combines:
        1. Set Cardinality: Normalized set size |C(x)| / C (higher -> more classes needed for coverage)
        2. Softmax Ambiguity within set: 1 - max_{c in C(x)} p(c|x)
        
        Returns:
            1D array of shape (N,) with conformal uncertainty scores. Higher = more uncertain.
        """
        if not self.is_calibrated:
            raise RuntimeError("SplitConformalPredictor must be calibrated to compute conformal uncertainty.")
            
        N, C = probs.shape
        pred_sets = self.predict_sets(probs)
        
        scores = np.zeros(N, dtype=np.float64)
        for i in range(N):
            c_set = pred_sets[i]
            k_set = len(c_set)
            
            # Primary signal: prediction set cardinality
            card_score = (k_set - 1) / max(1, C - 1)
            
            # Secondary signal: ambiguity among classes in the prediction set
            if k_set > 1:
                p_in_set = probs[i, c_set]
                sorted_p = np.sort(p_in_set)
                # margin between top two in set
                set_margin = sorted_p[-1] - sorted_p[-2]
                ambiguity = 1.0 - set_margin
            else:
                ambiguity = 1.0 - probs[i, c_set[0]]
                
            scores[i] = 0.6 * card_score + 0.4 * ambiguity
            
        return scores

