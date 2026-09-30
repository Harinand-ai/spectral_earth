"""
Unit verification for Expected Learning Value strategy.
"""
import numpy as np
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

from src.query_engine.query_engine import QueryEngine, ExpectedLearningValueController
from src.query_engine.query_engine.expected_learning_value import (
    compute_expected_gradient_signal,
    compute_representation_novelty,
)

def test_expected_learning_value_unit():
    print("=" * 60)
    print("RUNNING EXPECTED LEARNING VALUE UNIT VERIFICATION")
    print("=" * 60)

    np.random.seed(42)
    N = 100
    D = 2048
    C = 16
    k = 10

    embeddings = np.random.randn(N, D).astype(np.float64)
    probs = np.random.dirichlet(np.ones(C), size=N).astype(np.float64)
    labeled_emb = np.random.randn(20, D).astype(np.float64)

    # Test components
    g_signal = compute_expected_gradient_signal(probs, embeddings)
    assert len(g_signal) == N
    assert np.all(g_signal >= 0)
    print(f"Gradient signal range: [{np.min(g_signal):.4f}, {np.max(g_signal):.4f}]")

    novelty = compute_representation_novelty(embeddings, labeled_emb)
    assert len(novelty) == N
    assert np.all(novelty >= 0)
    print(f"Representation novelty range: [{np.min(novelty):.4f}, {np.max(novelty):.4f}]")

    # Test Controller
    controller = ExpectedLearningValueController()
    selected_idx, details = controller.select_samples(
        embeddings=embeddings,
        probabilities=probs,
        k=k,
        labeled_embeddings=labeled_emb,
    )
    assert len(selected_idx) == k
    print(f"Controller selected indices (k={k}): {selected_idx}")

    # Test QueryEngine integration
    qe = QueryEngine(uncertainty_method="expected_learning_value")
    qe_selected = qe.select(
        embeddings=embeddings,
        probabilities=probs,
        k=k,
        labeled_embeddings=labeled_emb,
    )
    assert len(qe_selected) == k
    assert np.array_equal(selected_idx, qe_selected)

    print("PASS: Expected Learning Value Unit Verification Passed Successfully!")

if __name__ == "__main__":
    test_expected_learning_value_unit()
