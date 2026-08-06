import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.backbones.registry import BACKBONE_REGISTRY
from src.learning_engine import (
    FeatureExtractor,
    ClassifierHead,
    ActiveLearningDatasetManager,
    ActiveLearningTrainer,
    LearningEngine,
)
from src.query_engine.query_engine import QueryEngine


class DummyDataset(torch.utils.data.Dataset):
    """Dummy Hyperspectral Patch Dataset for fast verification."""
    def __init__(self, num_samples=100, num_bands=200, patch_size=15, num_classes=5):
        self.num_samples = num_samples
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Synthetic patches [C, H, W]
        self.images = torch.randn(num_samples, num_bands, patch_size, patch_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
        }


def get_backbone_instance(name="spec_resnet50", num_classes=5, in_channels=200):
    backbone_cls = BACKBONE_REGISTRY[name]
    if "in_channels" in backbone_cls.__init__.__code__.co_varnames:
        return backbone_cls(num_classes=num_classes, in_channels=in_channels)
    else:
        return backbone_cls(num_classes=num_classes)


def test_frozen_backbone_and_trainable_head():
    print("\n--- TEST 1: Frozen Backbone & Trainable Head Verification ---")
    backbone = get_backbone_instance("spec_resnet50", num_classes=5, in_channels=200)

    feature_extractor = FeatureExtractor(backbone=backbone)
    classifier_head = ClassifierHead(in_features=feature_extractor.feature_dim, num_classes=5)

    # Verify all backbone parameters are frozen
    backbone_frozen = all(not p.requires_grad for p in feature_extractor.backbone.parameters())
    head_trainable = all(p.requires_grad for p in classifier_head.parameters())

    print(f"Backbone parameters frozen: {backbone_frozen}")
    print(f"Classifier head parameters trainable: {head_trainable}")

    assert backbone_frozen, "FAIL: Backbone parameters must be frozen!"
    assert head_trainable, "FAIL: Classifier head parameters must be trainable!"
    print("PASS: Test 1 Passed!")


def test_active_learning_loop():
    print("\n--- TEST 2: Active Learning Loop & Pool Updates Verification ---")
    
    # 1. Setup Dummy Data
    train_dataset = DummyDataset(num_samples=100, num_classes=5)
    test_dataset = DummyDataset(num_samples=30, num_classes=5)

    # 2. Dataset Manager
    dm = ActiveLearningDatasetManager(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        initial_labeled_ratio=0.10, # 10 labeled, 90 unlabeled
        seed=42,
    )

    initial_labeled_count = dm.num_labeled
    initial_unlabeled_count = dm.num_unlabeled
    print(f"Initial Pool Sizes: Labeled={initial_labeled_count}, Unlabeled={initial_unlabeled_count}")
    assert initial_labeled_count == 10
    assert initial_unlabeled_count == 90

    # 3. Model A Engine
    backbone = get_backbone_instance("spec_resnet50", num_classes=5, in_channels=200)

    engine = LearningEngine(
        backbone=backbone,
        num_classes=5,
        dataset_manager=dm,
        device="cpu",
    )

    # 4. Model B Query Engine
    qe = QueryEngine(uncertainty_method="entropy")

    # 5. Run AL loop for 3 rounds with k=10 per round
    history = engine.run_full_active_learning_loop(
        query_engine=qe,
        k=10,
        total_rounds=3,
        epochs_per_round=2,
        batch_size=16,
    )

    print("\nAL Loop Execution Summary:")
    for h in history:
        print(f"Round {h['round']} | Labeled: {h['num_labeled']} | Unlabeled: {h['num_unlabeled']} | Queried: {h['queried_samples']} | Loss: {h['retrain_loss']:.4f} | OA: {h.get('metrics', {}).get('OverallAccuracy', 0.0)*100:.2f}%")

    # Verify Pool Size transitions across 3 rounds
    assert history[0]["num_labeled"] == 10
    assert history[1]["num_labeled"] == 20
    assert history[2]["num_labeled"] == 30
    assert history[3]["num_labeled"] == 40
    print("PASS: Pool sizes updated precisely by k samples each round!")
    print("PASS: Test 2 Passed!")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING SPECTRALEARTH ACTIVE LEARNING INTEGRATION VERIFICATION")
    print("="*60)
    test_frozen_backbone_and_trainable_head()
    test_active_learning_loop()
    print("\nALL INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY!")
