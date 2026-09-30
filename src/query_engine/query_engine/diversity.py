from typing import Optional
import numpy as np


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Validate feature embeddings.

    Expected shape:
        (num_samples, embedding_dimension)
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape "
            "(num_samples, embedding_dimension)"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("embeddings cannot be empty")

    if embeddings.shape[1] == 0:
        raise ValueError("embedding dimension cannot be zero")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contain NaN or infinite values")

    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings.
    """
    embeddings = _validate_embeddings(embeddings)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)

    return embeddings / norms


def select_diverse_samples(
    embeddings: np.ndarray,
    k: int,
    start_index: Optional[int] = None,
    labeled_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Select K diverse samples using greedy farthest-first traversal.

    If labeled_embeddings is provided, the distance to existing labeled samples
    is used to initialize candidate distances, selecting samples that maximize
    distance to both existing labeled data and previously selected candidates.

    Otherwise, the first sample is chosen as the point farthest from the
    candidate embedding centroid unless start_index is supplied.

    Parameters
    ----------
    embeddings:
        Shape (num_samples, embedding_dimension)
    k:
        Number of samples to select.
    start_index:
        Optional starting sample index.
    labeled_embeddings:
        Optional shape (num_labeled, embedding_dimension) for labeled-pool-aware diversity.

    Returns
    -------
    np.ndarray
        Selected sample indices relative to embeddings.
    """

    embeddings = normalize_embeddings(embeddings)
    n_samples = embeddings.shape[0]

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > n_samples:
        raise ValueError("k cannot be greater than the number of samples")

    # --------------------------------------------------
    # Distance Initialization
    # --------------------------------------------------

    if labeled_embeddings is not None and len(labeled_embeddings) > 0:
        labeled_embeddings = normalize_embeddings(labeled_embeddings)
        # Compute pairwise distance matrix [n_samples, n_labeled]
        # dist(x, L) = min_j || x - L_j ||
        # For memory efficiency, compute in batches or dot product
        # ||x - l||^2 = 2 - 2 (x . l) for unit vectors
        sims = np.dot(embeddings, labeled_embeddings.T) # [n_samples, n_labeled]
        sims = np.clip(sims, -1.0, 1.0)
        # Euclidean dist = sqrt(2 * (1 - sim))
        dists_to_labeled = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - sims)))
        min_distances = np.min(dists_to_labeled, axis=1) # [n_samples]

        first_index = int(np.argmax(min_distances))
        selected = [first_index]
        new_dists = np.linalg.norm(embeddings - embeddings[first_index], axis=1)
        min_distances = np.minimum(min_distances, new_dists)
        min_distances[first_index] = -np.inf

    else:
        if start_index is None:
            centroid = np.mean(embeddings, axis=0, keepdims=True)
            distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
            first_index = int(np.argmax(distances_to_centroid))
        else:
            if start_index < 0 or start_index >= n_samples:
                raise ValueError("start_index is out of range")
            first_index = int(start_index)

        selected = [first_index]
        min_distances = np.linalg.norm(embeddings - embeddings[first_index], axis=1)
        min_distances[first_index] = -np.inf

    # --------------------------------------------------
    # Greedy farthest-first selection
    # --------------------------------------------------

    while len(selected) < k:
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)

        new_distances = np.linalg.norm(embeddings - embeddings[next_index], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected] = -np.inf

    return np.asarray(selected, dtype=np.int64)