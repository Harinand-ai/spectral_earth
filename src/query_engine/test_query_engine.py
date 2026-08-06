import numpy as np

from query_engine import QueryEngine


# =====================================================
# Dummy data representing Model A output
# =====================================================

embeddings = np.array([
    [1.00, 0.00],
    [0.95, 0.05],
    [0.00, 1.00],
    [0.05, 0.95],
    [-1.00, 0.00],
    [-0.95, 0.05],
])


probabilities = np.array([
    [0.90, 0.05, 0.05],
    [0.80, 0.10, 0.10],
    [0.34, 0.33, 0.33],
    [0.40, 0.35, 0.25],
    [0.70, 0.20, 0.10],
    [0.45, 0.40, 0.15],
])


# =====================================================
# Create Query Engine
# =====================================================

engine = QueryEngine(
    uncertainty_method="entropy"
)


# =====================================================
# Early active-learning round
# =====================================================

early_selected = engine.select(
    embeddings=embeddings,
    probabilities=probabilities,
    k=3,
    current_round=0,
    total_rounds=10,
)

print("===== EARLY ROUND =====")
print("Selected indices:", early_selected)


# =====================================================
# Late active-learning round
# =====================================================

late_selected = engine.select(
    embeddings=embeddings,
    probabilities=probabilities,
    k=3,
    current_round=9,
    total_rounds=10,
)

print("\n===== LATE ROUND =====")
print("Selected indices:", late_selected)


# =====================================================
# Validation
# =====================================================

assert len(early_selected) == 3
assert len(late_selected) == 3

assert len(np.unique(early_selected)) == 3
assert len(np.unique(late_selected)) == 3

assert np.all(early_selected >= 0)
assert np.all(late_selected >= 0)

print("\nMODEL B QUERY ENGINE TEST PASSED!")