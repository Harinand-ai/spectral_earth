import unittest
import numpy as np
import torch
import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.query_engine.query_engine.uncertainty import margin_uncertainty, compute_uncertainty
from src.query_engine.query_engine.engine import QueryEngine


class TestPureMarginSampling(unittest.TestCase):
    """
    Automated Validation and Test Suite for Pure Margin Sampling.
    Verifies mathematical fidelity of u(x) = 1 - (p_(1) - p_(2)),
    deterministic ranking, absence of diversity corruption, and QueryEngine integration.
    """

    def setUp(self):
        np.random.seed(42)

    def test_margin_uncertainty_exact_formula(self):
        """Verify u(x) = 1 - (p_(1) - p_(2)) on known synthetic probabilities."""
        probs = np.array([
            [0.90, 0.05, 0.05],  # p(1)=0.90, p(2)=0.05 -> margin=0.85 -> unc=0.15
            [0.51, 0.49, 0.00],  # p(1)=0.51, p(2)=0.49 -> margin=0.02 -> unc=0.98
            [0.34, 0.33, 0.33],  # p(1)=0.34, p(2)=0.33 -> margin=0.01 -> unc=0.99
            [0.70, 0.20, 0.10],  # p(1)=0.70, p(2)=0.20 -> margin=0.50 -> unc=0.50
        ])
        expected_unc = np.array([0.15, 0.98, 0.99, 0.50])
        computed_unc = margin_uncertainty(probs)

        np.testing.assert_allclose(computed_unc, expected_unc, atol=1e-6)

        # Also verify via compute_uncertainty
        computed_via_wrapper = compute_uncertainty(probs, method="margin")
        np.testing.assert_allclose(computed_via_wrapper, expected_unc, atol=1e-6)

    def test_query_engine_pure_margin_selection(self):
        """Verify QueryEngine selects highest uncertainty (smallest margin) samples."""
        probs = np.array([
            [0.90, 0.05, 0.05],  # unc=0.15 (rank 4)
            [0.51, 0.49, 0.00],  # unc=0.98 (rank 2)
            [0.34, 0.33, 0.33],  # unc=0.99 (rank 1)
            [0.70, 0.20, 0.10],  # unc=0.50 (rank 3)
        ])
        # Arbitrary embeddings that might otherwise alter ranking if diversity were active
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])

        qe = QueryEngine(uncertainty_method="margin")
        selected = qe.select(embeddings, probs, k=2)

        # Expected top 2: index 2 (unc=0.99) and index 1 (unc=0.98)
        self.assertEqual(list(selected), [2, 1])

        # Verify weights indicate 100% uncertainty, 0% diversity
        score_res = qe.score(embeddings, probs)
        self.assertEqual(score_res["uncertainty_weight"], 1.0)
        self.assertEqual(score_res["diversity_weight"], 0.0)
        np.testing.assert_allclose(score_res["final"], score_res["uncertainty"])

    def test_pure_margin_alias_compatibility(self):
        """Verify 'pure_margin' alias works identically to 'margin'."""
        probs = np.array([
            [0.60, 0.40],  # unc=0.80
            [0.99, 0.01],  # unc=0.02
        ])
        embeddings = np.random.randn(2, 64)

        qe = QueryEngine(uncertainty_method="pure_margin")
        selected = qe.select(embeddings, probs, k=1)
        self.assertEqual(selected[0], 0)

    def test_pure_margin_deterministic_tie_breaking(self):
        """Verify deterministic selection when margins are identical."""
        probs = np.array([
            [0.60, 0.40],  # margin = 0.20
            [0.60, 0.40],  # margin = 0.20
            [0.60, 0.40],  # margin = 0.20
        ])
        embeddings = np.ones((3, 10))

        qe = QueryEngine(uncertainty_method="margin")
        sel1 = qe.select(embeddings, probs, k=2, current_round=0)
        sel2 = qe.select(embeddings, probs, k=2, current_round=0)

        self.assertEqual(list(sel1), list(sel2))
        self.assertEqual(len(np.unique(sel1)), 2)


if __name__ == "__main__":
    unittest.main()

