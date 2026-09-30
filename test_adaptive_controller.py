"""
Unit test for AdaptiveQueryController and QueryEngine integration.
"""
import numpy as np
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.query_engine.query_engine import QueryEngine, AdaptiveQueryController

def test_controller_unit():
    print("============================================================")
    print("RUNNING ADAPTIVE QUERY CONTROLLER UNIT VERIFICATION")
    print("============================================================")

    N, D, C = 100, 2048, 16
    np.random.seed(42)
    embeddings = np.random.randn(N, D)
    logits = np.random.randn(N, C)
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    labeled_embeddings = embeddings[:10]
    unlabeled_embeddings = embeddings[10:]
    unlabeled_probs = probabilities[10:]

    controller = AdaptiveQueryController()

    # Test round 0
    weights_r0 = controller.compute_controller_weights(
        probabilities=unlabeled_probs,
        current_round=0,
        total_rounds=5,
        embeddings=unlabeled_embeddings,
        labeled_embeddings=labeled_embeddings,
    )
    print(f"Round 0 Controller Weights: {weights_r0}")
    assert np.isclose(weights_r0["w_uncertainty"] + weights_r0["w_diversity"] + weights_r0["w_hard_region"], 1.0)

    # Test round 4
    weights_r4 = controller.compute_controller_weights(
        probabilities=unlabeled_probs,
        current_round=4,
        total_rounds=5,
        embeddings=unlabeled_embeddings,
        labeled_embeddings=labeled_embeddings,
    )
    print(f"Round 4 Controller Weights: {weights_r4}")
    assert weights_r4["w_uncertainty"] > weights_r0["w_uncertainty"], "Uncertainty weight should increase in later rounds"

    # Test QueryEngine integration
    qe = QueryEngine(uncertainty_method="adaptive_controller")
    selected = qe.select(
        embeddings=unlabeled_embeddings,
        probabilities=unlabeled_probs,
        k=10,
        current_round=0,
        total_rounds=5,
        labeled_embeddings=labeled_embeddings,
    )
    print(f"Selected indices (k=10): {selected}")
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10

    print("PASS: AdaptiveQueryController Unit Verification Passed Successfully!")

if __name__ == "__main__":
    test_controller_unit()
