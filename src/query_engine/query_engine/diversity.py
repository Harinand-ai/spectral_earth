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
) -> np.ndarray:
    """
    Select K diverse samples using greedy farthest-first traversal.

    The first sample is chosen as the point farthest from the
    global embedding centroid unless start_index is supplied.

    Each following sample is the point whose minimum distance
    from all previously selected samples is largest.

    Parameters
    ----------
    embeddings:
        Shape (num_samples, embedding_dimension)

    k:
        Number of samples to select.

    start_index:
        Optional starting sample.

    Returns
    -------
    np.ndarray
        Selected sample indices.
    """

    embeddings = normalize_embeddings(embeddings)

    n_samples = embeddings.shape[0]

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > n_samples:
        raise ValueError(
            "k cannot be greater than the number of samples"
        )

    # --------------------------------------------------
    # Choose first representative sample
    # --------------------------------------------------

    if start_index is None:

        centroid = np.mean(
            embeddings,
            axis=0,
            keepdims=True,
        )

        distances_to_centroid = np.linalg.norm(
            embeddings - centroid,
            axis=1,
        )

        first_index = int(
            np.argmax(distances_to_centroid)
        )

    else:

        if start_index < 0 or start_index >= n_samples:
            raise ValueError("start_index is out of range")

        first_index = int(start_index)

    selected = [first_index]

    # Distance from every point to the selected set.
    min_distances = np.linalg.norm(
        embeddings - embeddings[first_index],
        axis=1,
    )

    min_distances[first_index] = -np.inf

    # --------------------------------------------------
    # Greedy farthest-first selection
    # --------------------------------------------------

    while len(selected) < k:

        next_index = int(
            np.argmax(min_distances)
        )

        selected.append(next_index)

        new_distances = np.linalg.norm(
            embeddings - embeddings[next_index],
            axis=1,
        )

        min_distances = np.minimum(
            min_distances,
            new_distances,
        )

        min_distances[selected] = -np.inf

    return np.asarray(
        selected,
        dtype=np.int64,
    )