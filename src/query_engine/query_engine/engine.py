from typing import Optional
import numpy as np

from .uncertainty import compute_uncertainty
from .adaptive import combine_scores, get_adaptive_weights


class QueryEngine:
    """
    Model B - Intelligent Query Engine.

    Combines:
        1. Uncertainty estimation
        2. Embedding-based diversity
        3. Adaptive uncertainty/diversity weighting
        4. Top-K query selection

    Higher scores always indicate higher query priority.
    """

    SUPPORTED_METHODS = (
        "entropy",
        "margin",
        "least_confidence",
        "diversity",
        "random",
        "adaptive",
        "hybrid",
        "lc_adaptive",
        "lc_hybrid",
        "filter_diversity",
        "filter_diversity_labeled",
        "adaptive_controller",
        "expected_learning_value",
        "value_of_information",
    )

    def __init__(
        self,
        uncertainty_method: str = "entropy",
        start_uncertainty: float = 0.30,
        end_uncertainty: float = 0.80,
    ):
        uncertainty_method = uncertainty_method.lower()

        if uncertainty_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported uncertainty method: "
                f"{uncertainty_method}. "
                f"Choose from {self.SUPPORTED_METHODS}"
            )

        if not 0.0 <= start_uncertainty <= 1.0:
            raise ValueError(
                "start_uncertainty must be between 0 and 1"
            )

        if not 0.0 <= end_uncertainty <= 1.0:
            raise ValueError(
                "end_uncertainty must be between 0 and 1"
            )

        self.uncertainty_method = uncertainty_method
        self.start_uncertainty = float(start_uncertainty)
        self.end_uncertainty = float(end_uncertainty)

    @staticmethod
    def _validate_inputs(
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        k: int,
    ):
        embeddings = np.asarray(
            embeddings,
            dtype=np.float64,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (N, D)"
            )

        if probabilities.ndim != 2:
            raise ValueError(
                "probabilities must have shape (N, C)"
            )

        if embeddings.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "embeddings and probabilities must contain "
                "the same number of samples"
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "Input pool cannot be empty"
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "embeddings contain NaN or infinite values"
            )

        if not isinstance(k, (int, np.integer)):
            raise TypeError(
                "k must be an integer"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        if k > embeddings.shape[0]:
            raise ValueError(
                "k cannot exceed the number of samples"
            )

        return embeddings, probabilities

    @staticmethod
    def compute_diversity_scores(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute one embedding-based diversity score
        for every candidate.

        Higher score = farther from the current feature centroid.
        """

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        normalized_embeddings = (
            embeddings / norms
        )

        centroid = np.mean(
            normalized_embeddings,
            axis=0,
            keepdims=True,
        )

        distances = np.linalg.norm(
            normalized_embeddings - centroid,
            axis=1,
        )

        return distances

    def score(
        self,
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        current_round: int = 0,
        total_rounds: int = 10,
    ) -> dict:
        """
        Calculate all Model-B scores without selecting samples.

        Useful for:
            - experiments
            - debugging
            - logging
            - research analysis
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float64,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape (N, D)"
            )

        if probabilities.ndim != 2:
            raise ValueError(
                "probabilities must have shape (N, C)"
            )

        if embeddings.shape[0] != probabilities.shape[0]:
            raise ValueError(
                "embeddings and probabilities must contain "
                "the same number of samples"
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                "Input pool cannot be empty"
            )

        if self.uncertainty_method == "random":
            N = len(embeddings)
            rng = np.random.RandomState(current_round + 42)
            scores = rng.rand(N)
            return {
                "uncertainty": scores,
                "diversity": scores,
                "final": scores,
                "uncertainty_weight": 0.5,
                "diversity_weight": 0.5,
            }

        diversity_scores = self.compute_diversity_scores(embeddings)

        if self.uncertainty_method == "diversity":
            return {
                "uncertainty": np.zeros_like(diversity_scores),
                "diversity": diversity_scores,
                "final": diversity_scores,
                "uncertainty_weight": 0.0,
                "diversity_weight": 1.0,
            }

        if self.uncertainty_method in {"hybrid", "lc_hybrid"}:
            unc_scores = compute_uncertainty(probabilities, method="least_confidence")
            from .adaptive import normalize_scores
            norm_unc = normalize_scores(unc_scores)
            norm_div = normalize_scores(diversity_scores)
            final_scores = 0.5 * norm_unc + 0.5 * norm_div
            return {
                "uncertainty": unc_scores,
                "diversity": diversity_scores,
                "final": final_scores,
                "uncertainty_weight": 0.5,
                "diversity_weight": 0.5,
            }

        # For adaptive, lc_adaptive, and default: use least_confidence unless explicitly requested otherwise
        if self.uncertainty_method in {"adaptive", "lc_adaptive"}:
            unc_method = "least_confidence"
        else:
            unc_method = self.uncertainty_method

        uncertainty_scores = compute_uncertainty(
            probabilities,
            method=unc_method,
        )

        final_scores = combine_scores(
            uncertainty_scores=uncertainty_scores,
            diversity_scores=diversity_scores,
            current_round=current_round,
            total_rounds=total_rounds,
            start_uncertainty=self.start_uncertainty,
            end_uncertainty=self.end_uncertainty,
        )

        uncertainty_weight, diversity_weight = (
            get_adaptive_weights(
                current_round=current_round,
                total_rounds=total_rounds,
                start_uncertainty=self.start_uncertainty,
                end_uncertainty=self.end_uncertainty,
            )
        )

        return {
            "uncertainty": uncertainty_scores,
            "diversity": diversity_scores,
            "final": final_scores,
            "uncertainty_weight": uncertainty_weight,
            "diversity_weight": diversity_weight,
        }

    def select(
        self,
        embeddings: np.ndarray,
        probabilities: np.ndarray,
        k: int,
        current_round: int = 0,
        total_rounds: int = 10,
        labeled_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Main integration interface with tie handling stability.

        Returns
        -------
        np.ndarray
            Selected indices relative to the input pool.
        """

        embeddings, probabilities = (
            self._validate_inputs(
                embeddings,
                probabilities,
                k,
            )
        )

        if self.uncertainty_method in {"expected_learning_value", "value_of_information"}:
            from .expected_learning_value import ExpectedLearningValueController
            if self.uncertainty_method == "value_of_information":
                controller = ExpectedLearningValueController(w_uncertainty=0.5, w_novelty=0.5, w_gradient=0.0)
            else:
                controller = ExpectedLearningValueController(w_uncertainty=0.4, w_novelty=0.4, w_gradient=0.2)
            
            selected_indices, _ = controller.select_samples(
                embeddings=embeddings,
                probabilities=probabilities,
                k=k,
                labeled_embeddings=labeled_embeddings,
            )
            return selected_indices.astype(np.int64)

        if self.uncertainty_method == "adaptive_controller":
            from .adaptive_controller import AdaptiveQueryController
            controller = AdaptiveQueryController()
            return controller.select(
                embeddings=embeddings,
                probabilities=probabilities,
                k=k,
                current_round=current_round,
                total_rounds=total_rounds,
                labeled_embeddings=labeled_embeddings,
            )

        if self.uncertainty_method in {"filter_diversity", "filter_diversity_labeled"}:
            from .uncertainty import least_confidence
            from .diversity import select_diverse_samples

            unc_scores = least_confidence(probabilities)
            cand_pool_size = min(len(embeddings), max(4 * k, 50))
            top_cand_rel_indices = np.argsort(-unc_scores, kind="stable")[:cand_pool_size]
            cand_embeddings = embeddings[top_cand_rel_indices]

            if self.uncertainty_method == "filter_diversity_labeled":
                selected_in_cand = select_diverse_samples(
                    cand_embeddings, k, labeled_embeddings=labeled_embeddings
                )
            else:
                selected_in_cand = select_diverse_samples(cand_embeddings, k)

            selected_indices = top_cand_rel_indices[selected_in_cand]
            return selected_indices.astype(np.int64)

        results = self.score(
            embeddings=embeddings,
            probabilities=probabilities,
            current_round=current_round,
            total_rounds=total_rounds,
        )

        final_scores = results["final"]

        # Deterministic tie-breaking noise for stable ranking reproducibility
        rng = np.random.RandomState(current_round + 1000)
        tie_breaker = rng.uniform(0, 1e-9, size=len(final_scores))
        stable_scores = final_scores + tie_breaker

        selected_indices = np.argsort(-stable_scores, kind="stable")[:k]

        return selected_indices.astype(
            np.int64
        )

    def __repr__(self):
        return (
            "QueryEngine("
            f"uncertainty_method='{self.uncertainty_method}', "
            f"start_uncertainty={self.start_uncertainty}, "
            f"end_uncertainty={self.end_uncertainty}"
            ")"
        )