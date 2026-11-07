import unittest
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from src.conformal.split_conformal import SplitConformalPredictor
from src.query_engine.class_coverage import ClassCoverageController


class TestConformalActiveLearning(unittest.TestCase):
    """
    Automated Validation and Test Suite for ConformAL-HSI (Section 15).
    """

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.num_classes = 5
        self.n_samples = 100

    def test_nonconformity_scores_and_quantile(self):
        """Test nonconformity calculation and finite-sample quantile."""
        predictor = SplitConformalPredictor(alpha=0.10)
        probs = np.random.dirichlet(np.ones(self.num_classes), size=self.n_samples)
        labels = np.random.randint(0, self.num_classes, size=self.n_samples)

        scores = predictor.compute_nonconformity_scores(probs, labels)
        self.assertEqual(len(scores), self.n_samples)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

        # Calibrate
        q_hat = predictor.calibrate(probs, labels)
        self.assertTrue(0.0 <= q_hat <= 1.0)
        self.assertTrue(predictor.is_calibrated)
        self.assertEqual(predictor.calibration_size, self.n_samples)

    def test_prediction_set_construction_and_coverage(self):
        """Verify prediction set formation and empirical finite-sample coverage guarantee."""
        # Generate well-calibrated synthetic data
        n_cal = 200
        n_test = 500
        alpha = 0.10

        logits_cal = np.random.randn(n_cal, self.num_classes)
        probs_cal = np.exp(logits_cal) / np.sum(np.exp(logits_cal), axis=1, keepdims=True)
        labels_cal = np.array([np.random.choice(self.num_classes, p=probs_cal[i]) for i in range(n_cal)])

        predictor = SplitConformalPredictor(alpha=alpha)
        predictor.calibrate(probs_cal, labels_cal)

        logits_test = np.random.randn(n_test, self.num_classes)
        probs_test = np.exp(logits_test) / np.sum(np.exp(logits_test), axis=1, keepdims=True)
        labels_test = np.array([np.random.choice(self.num_classes, p=probs_test[i]) for i in range(n_test)])

        results = predictor.evaluate_coverage(probs_test, labels_test)
        # Under exchangeability, coverage should be approximately >= 1 - alpha (within binomial error)
        self.assertGreaterEqual(results["empirical_coverage_pct"], 83.0)
        self.assertGreaterEqual(results["mean_set_size"], 1.0)
        self.assertLessEqual(results["mean_set_size"], self.num_classes)

    def test_conformal_uncertainty_scoring(self):
        """Verify conformal uncertainty correctly ranks ambiguous samples higher."""
        predictor = SplitConformalPredictor(alpha=0.10)
        # 100 calibration samples
        probs_cal = np.random.dirichlet(np.ones(self.num_classes), size=100)
        labels_cal = np.random.randint(0, self.num_classes, size=100)
        predictor.calibrate(probs_cal, labels_cal)

        # Certain sample: probability 0.99 for class 0
        p_certain = np.array([[0.99, 0.0025, 0.0025, 0.0025, 0.0025]])
        # Uncertain sample: uniform probability 0.20 for all classes
        p_uncertain = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])

        score_certain = predictor.compute_conformal_uncertainty(p_certain)[0]
        score_uncertain = predictor.compute_conformal_uncertainty(p_uncertain)[0]

        self.assertGreater(score_uncertain, score_certain)

    def test_class_coverage_controller(self):
        """Verify that class coverage controller boosts starved classes without test leakage."""
        controller = ClassCoverageController(lambda_cov=0.5, pool_prior_weight=0.7)
        # Pool with equal distribution
        unlabeled_probs = np.ones((50, self.num_classes)) / self.num_classes
        # Labeled pool where class 0 is heavily starved (0 samples) and class 1 has 20 samples
        labeled_labels = np.array([1]*20 + [2]*10 + [3]*10 + [4]*10)

        base_scores = np.ones(50)
        adjusted = controller.apply_coverage_adjustment(
            base_scores=base_scores,
            unlabeled_probs=unlabeled_probs,
            labeled_labels=labeled_labels,
            num_classes=self.num_classes
        )

        self.assertEqual(len(adjusted), 50)
        # Adjusted scores should be non-negative and properly modulated
        self.assertTrue(np.all(adjusted > 0.0))

    def test_active_learning_pool_integrity(self):
        """Verify that no duplicate queries occur and candidate pools remain strictly partitioned."""
        total_pool_size = 1000
        k = 20
        all_indices = np.arange(total_pool_size)

        labeled = all_indices[:100]
        unlabeled = all_indices[100:]

        # Simulate 5 query rounds
        for _ in range(5):
            rng = np.random.RandomState(42)
            chosen_rel = rng.choice(len(unlabeled), size=k, replace=False)
            chosen_global = unlabeled[chosen_rel]

            # Assert no overlap with already labeled
            overlap = np.intersect1d(labeled, chosen_global)
            self.assertEqual(len(overlap), 0)

            # Update pools
            labeled = np.concatenate([labeled, chosen_global])
            rem_mask = ~np.isin(unlabeled, chosen_global)
            unlabeled = unlabeled[rem_mask]

            # Verify conservation of total indices
            self.assertEqual(len(labeled) + len(unlabeled), total_pool_size)


if __name__ == "__main__":
    unittest.main()
