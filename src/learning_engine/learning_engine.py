

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import numpy as np

from .feature_extractor import FeatureExtractor
from .classifier import ClassifierHead
from .dataset_manager import ActiveLearningDatasetManager
from .trainer import ActiveLearningTrainer

# Model B integration via Python import
from src.query_engine.query_engine import QueryEngine


class LearningEngine:
    """
    Model A - Learning Engine for Active Learning Framework.
    
    Responsibilities:
    - Maintains frozen SpectralEarth backbone (FeatureExtractor)
    - Manages trainable ClassifierHead
    - Manages Labeled and Unlabeled pool splits (ActiveLearningDatasetManager)
    - Executes training, retraining, inference, and evaluation (ActiveLearningTrainer)
    - Interacts with Model B (QueryEngine) via clean function calls:
      embeddings, probabilities -> QueryEngine.select -> selected indices -> update pools -> retrain
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dataset_manager: ActiveLearningDatasetManager,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 0,
        device: Optional[str] = None,
        multi_label: bool = False,
    ):
        self.num_classes = num_classes
        self.dataset_manager = dataset_manager
        self.multi_label = multi_label

        # Initialize Model A modules
        self.feature_extractor = FeatureExtractor(backbone=backbone, feature_dim=feature_dim)
        
        actual_feat_dim = self.feature_extractor.feature_dim
        self.classifier_head = ClassifierHead(
            in_features=actual_feat_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

        self.trainer = ActiveLearningTrainer(device=device, multi_label=multi_label)
        self.history: List[Dict[str, Any]] = []

    def initialize_classifier(
        self,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
    ) -> float:
        """
        Initial training of the classifier head on the seed labeled pool.
        """
        labeled_loader = self.dataset_manager.get_labeled_dataloader(batch_size=batch_size, shuffle=True)
        loss = self.trainer.train_classifier_head(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            labeled_dataloader=labeled_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        return loss

    def evaluate_test(self, batch_size: int = 64) -> Dict[str, Any]:
        """
        Evaluate current classifier head + frozen backbone on test dataset.
        """
        test_loader = self.dataset_manager.get_test_dataloader(batch_size=batch_size)
        if test_loader is None:
            return {}
        return self.trainer.evaluate(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            dataloader=test_loader,
            num_classes=self.num_classes,
        )

    def run_active_learning_step(
        self,
        query_engine: QueryEngine,
        k: int,
        current_round: int,
        total_rounds: int,
        epochs_per_round: int = 15,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Execute one active learning round:
        1. Extract embeddings and prediction probabilities for Unlabeled Pool (U).
        2. Call Model B (Query Engine) to select top-k informative samples.
        3. Update Labeled (L) and Unlabeled (U) dataset pools.
        4. Retrain ONLY classifier head on updated Labeled Pool.
        5. Evaluate updated model on Test Set.
        """
        unlabeled_loader = self.dataset_manager.get_unlabeled_dataloader(batch_size=batch_size)
        
        # 1. Feature extraction & prediction probabilities
        embeddings, probabilities = self.trainer.extract_unlabeled_embeddings_and_probs(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            unlabeled_dataloader=unlabeled_loader,
        )

        # 2. Query Engine selection (Model B)
        k_actual = min(k, len(embeddings))
        selected_relative_indices = query_engine.select(
            embeddings=embeddings,
            probabilities=probabilities,
            k=k_actual,
            current_round=current_round,
            total_rounds=total_rounds,
        )

        # 3. Pool updates
        added_global_indices = self.dataset_manager.update_pools(selected_relative_indices)

        # 4. Retrain classifier head on updated Labeled Pool
        labeled_loader = self.dataset_manager.get_labeled_dataloader(batch_size=batch_size, shuffle=True)
        retrain_loss = self.trainer.train_classifier_head(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            labeled_dataloader=labeled_loader,
            epochs=epochs_per_round,
            lr=lr,
        )

        # 5. Evaluate
        test_metrics = self.evaluate_test(batch_size=batch_size)

        step_summary = {
            "round": current_round + 1,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "queried_samples": len(added_global_indices),
            "retrain_loss": retrain_loss,
            "metrics": test_metrics,
        }

        self.history.append(step_summary)
        return step_summary

    def run_full_active_learning_loop(
        self,
        query_engine: QueryEngine,
        k: int,
        total_rounds: int,
        epochs_per_round: int = 15,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> List[Dict[str, Any]]:
        """
        Run complete multi-round Active Learning experiment.
        """
        # Initial training on seed pool
        init_loss = self.initialize_classifier(epochs=epochs_per_round, lr=lr, batch_size=batch_size)
        init_metrics = self.evaluate_test(batch_size=batch_size)

        self.history = [{
            "round": 0,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "queried_samples": 0,
            "retrain_loss": init_loss,
            "metrics": init_metrics,
        }]

        for r in range(1, total_rounds + 1):
            if self.dataset_manager.num_unlabeled == 0:
                break
            self.run_active_learning_step(
                query_engine=query_engine,
                k=k,
                current_round=r - 1,
                total_rounds=total_rounds,
                epochs_per_round=epochs_per_round,
                lr=lr,
                batch_size=batch_size,
            )

        return self.history

    def save_checkpoint(self, path: str):
        """Save state of classifier head, dataset manager pool indices, and run history."""
        state = {
            "classifier_head": self.classifier_head.state_dict(),
            "labeled_indices": self.dataset_manager.labeled_indices.tolist(),
            "unlabeled_indices": self.dataset_manager.unlabeled_indices.tolist(),
            "history": self.history,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load state of classifier head, dataset manager pool indices, and run history."""
        state = torch.load(path, map_location=self.trainer.device, weights_only=False)
        self.classifier_head.load_state_dict(state["classifier_head"])
        self.dataset_manager.labeled_indices = np.array(state["labeled_indices"], dtype=np.int64)
        self.dataset_manager.unlabeled_indices = np.array(state["unlabeled_indices"], dtype=np.int64)
        self.history = state.get("history", [])

