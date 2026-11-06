import unittest
import numpy as np
import torch
import torch.nn as nn

import pyrootutils
pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.learning_engine.pseudo_labeler import ConfidencePseudoLabeler
from src.learning_engine.classifier import ClassifierHead
from run_pseudolabel_controlled_ablation import train_classifier


class TestPseudoLabelingAndLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.device = "cpu"
        self.num_classes = 16
        self.head = ClassifierHead(
            in_features=64,
            num_classes=self.num_classes,
            hidden_dim=32,
            dropout=0.0,
            head_type="mlp_2layer",
            norm_layer="layernorm",
        )

    def test_weighted_loss_formula_mixed_batch(self):
        """Verify the weighted loss matches the per-sample mean formulation:
           L_batch = (sample_weights * per_sample_losses).mean()
                   = (sum_{gen} 1.0 * l_i + sum_{pseudo} lambda_p * l_j) / (N_gen + N_pseudo)
        """
        N_gen = 10
        N_pseudo = 6
        lambda_p = 0.2
        crit_none = nn.CrossEntropyLoss(reduction="none")

        x_gen = torch.randn(N_gen, 64)
        y_gen = torch.randint(0, self.num_classes, (N_gen,))
        x_pseudo = torch.randn(N_pseudo, 64)
        y_pseudo = torch.randint(0, self.num_classes, (N_pseudo,))

        with torch.no_grad():
            self.head.eval()
            logits_gen = self.head(x_gen)
            logits_pseudo = self.head(x_pseudo)
            l_gen = crit_none(logits_gen, y_gen)
            l_pseudo = crit_none(logits_pseudo, y_pseudo)

            expected_loss = (l_gen.sum() + lambda_p * l_pseudo.sum()) / (N_gen + N_pseudo)

            # Combined tensor pass
            x_cat = torch.cat([x_gen, x_pseudo], dim=0)
            y_cat = torch.cat([y_gen, y_pseudo], dim=0)
            is_pseudo = torch.tensor([False] * N_gen + [True] * N_pseudo, dtype=torch.bool)

            logits_cat = self.head(x_cat)
            sample_losses = crit_none(logits_cat, y_cat)
            sample_weights = torch.where(
                is_pseudo,
                torch.as_tensor(lambda_p, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )
            computed_loss = (sample_weights * sample_losses).mean()

            self.assertAlmostEqual(computed_loss.item(), expected_loss.item(), places=5)

    def test_weighted_loss_pure_genuine_batch(self):
        """A batch containing only genuine samples has weights 1.0, so loss.mean() == CrossEntropyLoss(reduction='mean')."""
        N_gen = 20
        crit_none = nn.CrossEntropyLoss(reduction="none")
        crit_mean = nn.CrossEntropyLoss(reduction="mean")

        x_gen = torch.randn(N_gen, 64)
        y_gen = torch.randint(0, self.num_classes, (N_gen,))
        is_pseudo = torch.tensor([False] * N_gen, dtype=torch.bool)

        with torch.no_grad():
            self.head.eval()
            logits = self.head(x_gen)
            std_loss = crit_mean(logits, y_gen)

            sample_losses = crit_none(logits, y_gen)
            sample_weights = torch.where(
                is_pseudo,
                torch.as_tensor(0.2, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )
            weighted_loss = (sample_weights * sample_losses).mean()

            self.assertAlmostEqual(weighted_loss.item(), std_loss.item(), places=5)

    def test_weighted_loss_pure_pseudo_batch(self):
        """A batch containing only pseudo samples has weights 0.2, so loss.mean() == 0.2 * CrossEntropyLoss(reduction='mean')."""
        N_pseudo = 20
        crit_none = nn.CrossEntropyLoss(reduction="none")
        crit_mean = nn.CrossEntropyLoss(reduction="mean")

        x_p = torch.randn(N_pseudo, 64)
        y_p = torch.randint(0, self.num_classes, (N_pseudo,))
        is_pseudo = torch.tensor([True] * N_pseudo, dtype=torch.bool)

        with torch.no_grad():
            self.head.eval()
            logits = self.head(x_p)
            std_loss = crit_mean(logits, y_p)

            sample_losses = crit_none(logits, y_p)
            sample_weights = torch.where(
                is_pseudo,
                torch.as_tensor(0.2, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )
            weighted_loss = (sample_weights * sample_losses).mean()

            # In a pure pseudo batch, every sample contributes 0.2 times the genuine loss
            self.assertAlmostEqual(weighted_loss.item(), (0.2 * std_loss).item(), places=5)

    def test_train_classifier_runs_and_decreases_loss(self):
        """Verify train_classifier executes successfully with mixed genuine & pseudo labels."""
        genuine_feats = torch.randn(30, 64)
        genuine_lbls = torch.randint(0, self.num_classes, (30,))
        pseudo_feats = torch.randn(15, 64)
        pseudo_lbls = torch.randint(0, self.num_classes, (15,))

        loss = train_classifier(
            self.head,
            genuine_feats,
            genuine_lbls,
            pseudo_feats=pseudo_feats,
            pseudo_lbls=pseudo_lbls,
            pseudo_weight=0.2,
            epochs=5,
            lr=0.005,
            device="cpu",
        )
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)

    def test_confidence_threshold_filtering(self):
        """Ensure samples with confidence < tau are rejected."""
        labeler = ConfidencePseudoLabeler(confidence_threshold=0.95, max_pseudo_per_class=15)
        # Create probabilities: 3 classes, 4 samples
        # Sample 0: max prob = 0.98 (class 1) -> ACCEPT
        # Sample 1: max prob = 0.90 (class 0) -> REJECT (<0.95)
        # Sample 2: max prob = 0.96 (class 2) -> ACCEPT
        # Sample 3: max prob = 0.85 (class 1) -> REJECT (<0.95)
        probs = np.array([
            [0.01, 0.98, 0.01],
            [0.90, 0.05, 0.05],
            [0.02, 0.02, 0.96],
            [0.10, 0.85, 0.05],
        ], dtype=np.float32)
        unlabeled_indices = np.array([100, 101, 102, 103], dtype=np.int64)

        res = labeler.select_pseudo_labels(probs, unlabeled_indices, num_classes=3)
        self.assertEqual(res["round_pseudo_count"], 2)
        np.testing.assert_array_equal(res["pseudo_indices"], np.array([100, 102]))
        np.testing.assert_array_equal(res["pseudo_labels"], np.array([1, 2]))

    def test_per_class_cap_enforcement(self):
        """Ensure no class exceeds max_pseudo_per_class, selecting the highest-confidence ones."""
        cap = 3
        labeler = ConfidencePseudoLabeler(confidence_threshold=0.90, max_pseudo_per_class=cap)
        # 6 samples all predicted as class 0 with varying confidences above 0.90
        probs = np.zeros((6, 2), dtype=np.float32)
        confs = [0.99, 0.92, 0.98, 0.95, 0.91, 0.96]
        for i, c in enumerate(confs):
            probs[i, 0] = c
            probs[i, 1] = 1.0 - c

        unlabeled_indices = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
        res = labeler.select_pseudo_labels(probs, unlabeled_indices, num_classes=2)

        # Expected top 3: 0.99 (idx 10), 0.98 (idx 12), 0.96 (idx 15)
        self.assertEqual(res["round_pseudo_count"], cap)
        self.assertEqual(res["per_class_counts"][0], cap)
        self.assertEqual(res["per_class_counts"][1], 0)
        self.assertSetEqual(set(res["pseudo_indices"].tolist()), {10, 12, 15})

    def test_duplicate_and_labeled_pool_exclusion(self):
        """Ensure samples present in genuine labeled pool are excluded and internal duplicates are stripped."""
        labeler = ConfidencePseudoLabeler(confidence_threshold=0.90, max_pseudo_per_class=10)
        probs = np.array([
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
        ], dtype=np.float32)
        # Suppose index 200 is already genuine labeled, and indices contain a duplicate 201
        unlabeled_indices = np.array([200, 201, 201], dtype=np.int64)
        labeled_indices = np.array([200, 50, 51], dtype=np.int64)

        res = labeler.select_pseudo_labels(
            probs, unlabeled_indices, num_classes=2, labeled_indices=labeled_indices
        )

        # 200 should be excluded because it's in labeled_indices
        # 201 should appear only once despite appearing twice in candidate array
        self.assertEqual(res["round_pseudo_count"], 1)
        self.assertEqual(res["pseudo_indices"][0], 201)
        self.assertEqual(res["per_class_counts"][0], 1)
        self.assertEqual(res["per_class_counts"][1], 0)

    def test_bookkeeping_and_cumulative_counts(self):
        """Ensure round_pseudo_count and cumulative_pseudo_count are properly separated across rounds,
           and reset_history clears cumulative tracking.
        """
        labeler = ConfidencePseudoLabeler(confidence_threshold=0.90, max_pseudo_per_class=10)

        # Round 1: samples 301, 302 accepted
        p1 = np.array([[0.95, 0.05], [0.96, 0.04]])
        u1 = np.array([301, 302])
        r1 = labeler.select_pseudo_labels(p1, u1, num_classes=2)
        self.assertEqual(r1["round_pseudo_count"], 2)
        self.assertEqual(r1["cumulative_pseudo_count"], 2)

        # Round 2: samples 302, 303 accepted (302 is seen previously, 303 is new)
        p2 = np.array([[0.97, 0.03], [0.98, 0.02]])
        u2 = np.array([302, 303])
        r2 = labeler.select_pseudo_labels(p2, u2, num_classes=2)
        self.assertEqual(r2["round_pseudo_count"], 2)
        # Cumulative unique samples across rounds: {301, 302, 303} -> exactly 3!
        self.assertEqual(r2["cumulative_pseudo_count"], 3)

        # Test reset
        labeler.reset_history()
        self.assertEqual(len(labeler.seen_pseudo_indices), 0)
        self.assertEqual(len(labeler.round_history), 0)

        # Next selection after reset starts fresh
        r3 = labeler.select_pseudo_labels(p1, u1, num_classes=2)
        self.assertEqual(r3["cumulative_pseudo_count"], 2)


if __name__ == "__main__":
    unittest.main()
