

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
        dropout: float = 0.0,
        head_type: str = "mlp_2layer",
        norm_layer: str = "layernorm",
        device: Optional[str] = None,
        multi_label: bool = False,
        use_class_weights: bool = False,
        use_pseudo_labeling: bool = False,
        pseudo_threshold: float = 0.95,
        pseudo_weight: float = 0.2,
        max_pseudo_per_class: int = 15,
    ):
        self.num_classes = num_classes
        self.dataset_manager = dataset_manager
        self.multi_label = multi_label
        self.use_class_weights = use_class_weights
        self.use_pseudo_labeling = use_pseudo_labeling
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_weight = pseudo_weight
        self.max_pseudo_per_class = max_pseudo_per_class

        from .pseudo_labeler import ConfidencePseudoLabeler
        self.pseudo_labeler = ConfidencePseudoLabeler(
            confidence_threshold=pseudo_threshold,
            max_pseudo_per_class=max_pseudo_per_class,
            pseudo_weight=pseudo_weight,
        )

        # Initialize Model A modules
        self.feature_extractor = FeatureExtractor(backbone=backbone, feature_dim=feature_dim)
        
        actual_feat_dim = self.feature_extractor.feature_dim
        self.classifier_head = ClassifierHead(
            in_features=actual_feat_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head_type=head_type,
            norm_layer=norm_layer,
        )

        self.trainer = ActiveLearningTrainer(device=device, multi_label=multi_label)
        self.history: List[Dict[str, Any]] = []

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        if not self.use_class_weights:
            return None
        labeled_dataset = self.dataset_manager.get_labeled_dataset()
        y_labeled = []
        for i in range(len(labeled_dataset)):
            item = labeled_dataset[i]
            lbl = item["label"] if isinstance(item, dict) else item[1]
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            y_labeled.append(lbl)
        counts = np.bincount(y_labeled, minlength=self.num_classes).astype(np.float32)
        total = float(len(y_labeled))
        weights = total / (self.num_classes * np.maximum(counts, 1.0))
        return torch.tensor(weights, dtype=torch.float32)

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
        class_weights = self._compute_class_weights()
        loss = self.trainer.train_classifier_head(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            labeled_dataloader=labeled_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=class_weights,
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
        import time
        round_start = time.time()
        unlabeled_loader = self.dataset_manager.get_unlabeled_dataloader(batch_size=batch_size)
        
        # 1. Feature extraction & prediction probabilities
        t_q_start = time.time()
        embeddings, probabilities = self.trainer.extract_unlabeled_embeddings_and_probs(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            unlabeled_dataloader=unlabeled_loader,
        )

        labeled_embeddings = None
        if query_engine.uncertainty_method == "filter_diversity_labeled":
            labeled_loader = self.dataset_manager.get_labeled_dataloader(batch_size=batch_size, shuffle=False)
            labeled_embeddings, _ = self.trainer.extract_unlabeled_embeddings_and_probs(
                feature_extractor=self.feature_extractor,
                classifier_head=self.classifier_head,
                unlabeled_dataloader=labeled_loader,
            )

        # 2. Query Engine selection (Model B)
        k_actual = min(k, len(embeddings))
        selected_relative_indices = query_engine.select(
            embeddings=embeddings,
            probabilities=probabilities,
            k=k_actual,
            current_round=current_round,
            total_rounds=total_rounds,
            labeled_embeddings=labeled_embeddings,
        )
        query_time = time.time() - t_q_start

        # 3. Pool updates
        added_global_indices = self.dataset_manager.update_pools(selected_relative_indices)

        # 4. Retrain classifier head on updated Labeled Pool
        t_tr_start = time.time()
        labeled_loader = self.dataset_manager.get_labeled_dataloader(batch_size=batch_size, shuffle=True)
        class_weights = self._compute_class_weights()
        retrain_loss = self.trainer.train_classifier_head(
            feature_extractor=self.feature_extractor,
            classifier_head=self.classifier_head,
            labeled_dataloader=labeled_loader,
            epochs=epochs_per_round,
            lr=lr,
            class_weights=class_weights,
        )

        pseudo_stats = {"count": 0, "per_class": {}, "mean_confidence": 0.0}
        if self.use_pseudo_labeling and self.dataset_manager.num_unlabeled > 0:
            rem_loader = self.dataset_manager.get_unlabeled_dataloader(batch_size=batch_size)
            rem_embeds, rem_probs = self.trainer.extract_unlabeled_embeddings_and_probs(
                feature_extractor=self.feature_extractor,
                classifier_head=self.classifier_head,
                unlabeled_dataloader=rem_loader,
            )
            p_res = self.pseudo_labeler.select_pseudo_labels(
                probabilities=rem_probs,
                unlabeled_indices=self.dataset_manager.unlabeled_indices,
                num_classes=self.num_classes,
                labeled_indices=self.dataset_manager.labeled_indices,
            )
            if p_res["total_accepted"] > 0:
                class PseudoDataset(torch.utils.data.Dataset):
                    def __init__(self, base_ds, g_indices, p_lbls):
                        self.base_ds = base_ds
                        self.g_indices = g_indices
                        self.p_lbls = p_lbls
                    def __len__(self):
                        return len(self.g_indices)
                    def __getitem__(self, i):
                        it = self.base_ds[self.g_indices[i]]
                        img = it["image"] if isinstance(it, dict) else it[0]
                        return {"image": img, "label": int(self.p_lbls[i])}

                pseudo_ds = PseudoDataset(self.dataset_manager.train_dataset, p_res["pseudo_indices"], p_res["pseudo_labels"])
                pseudo_loader = torch.utils.data.DataLoader(pseudo_ds, batch_size=batch_size, shuffle=True)
                retrain_loss = self.trainer.train_classifier_head(
                    feature_extractor=self.feature_extractor,
                    classifier_head=self.classifier_head,
                    labeled_dataloader=labeled_loader,
                    epochs=epochs_per_round,
                    lr=lr,
                    class_weights=class_weights,
                    pseudo_dataloader=pseudo_loader,
                    pseudo_weight=self.pseudo_weight,
                )
                pseudo_stats = {
                    "count": p_res["total_accepted"],
                    "per_class": p_res["per_class_counts"],
                    "mean_confidence": p_res["mean_confidence"],
                }

        train_time = time.time() - t_tr_start

        # 5. Evaluate
        test_metrics = self.evaluate_test(batch_size=batch_size)
        round_duration = time.time() - round_start

        step_summary = {
            "round": current_round + 1,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "num_pseudo": pseudo_stats["count"],
            "pseudo_stats": pseudo_stats,
            "queried_samples": len(added_global_indices),
            "retrain_loss": retrain_loss,
            "query_time": float(query_time),
            "train_time": float(train_time),
            "round_duration": float(round_duration),
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
        early_stopping_patience: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run complete multi-round Active Learning experiment with optional early stopping.
        """
        import time
        t_start = time.time()
        # Initial training on seed pool
        init_loss = self.initialize_classifier(epochs=epochs_per_round, lr=lr, batch_size=batch_size)
        init_metrics = self.evaluate_test(batch_size=batch_size)

        self.history = [{
            "round": 0,
            "num_labeled": self.dataset_manager.num_labeled,
            "num_unlabeled": self.dataset_manager.num_unlabeled,
            "queried_samples": 0,
            "retrain_loss": init_loss,
            "query_time": 0.0,
            "train_time": float(time.time() - t_start),
            "round_duration": float(time.time() - t_start),
            "metrics": init_metrics,
        }]

        no_improve_rounds = 0
        best_oa = init_metrics.get("OverallAccuracy", 0.0)

        for r in range(1, total_rounds + 1):
            if self.dataset_manager.num_unlabeled == 0:
                break
            step_res = self.run_active_learning_step(
                query_engine=query_engine,
                k=k,
                current_round=r - 1,
                total_rounds=total_rounds,
                epochs_per_round=epochs_per_round,
                lr=lr,
                batch_size=batch_size,
            )

            current_oa = step_res.get("metrics", {}).get("OverallAccuracy", 0.0)
            if early_stopping_patience is not None:
                if current_oa > best_oa + 1e-4:
                    best_oa = current_oa
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= early_stopping_patience:
                        print(f"Early stopping triggered at round {r} (patience={early_stopping_patience})")
                        break

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

