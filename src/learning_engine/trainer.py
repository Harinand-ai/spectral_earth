import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Any, Optional
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
)

from .feature_extractor import FeatureExtractor
from .classifier import ClassifierHead


class ActiveLearningTrainer:
    """
    Trainer for Model A (Learning Engine).
    
    Handles:
    1. Training/retraining ONLY the classifier head on the labeled pool.
    2. Extracting feature embeddings and prediction probabilities for the unlabeled pool.
    3. Evaluating performance on validation/test sets with comprehensive metrics.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        multi_label: bool = False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.multi_label = multi_label

    def train_classifier_head(
        self,
        feature_extractor: FeatureExtractor,
        classifier_head: ClassifierHead,
        labeled_dataloader: DataLoader,
        epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        reset_head: bool = False,
    ) -> float:
        """
        Train/retrain the classifier head while keeping feature_extractor strictly frozen.
        
        Returns:
            Average training loss in the final epoch.
        """
        feature_extractor.to(self.device)
        feature_extractor.eval()
        classifier_head.to(self.device)

        if reset_head:
            for layer in classifier_head.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        classifier_head.train()

        # Ensure feature_extractor parameters have no gradients
        for param in feature_extractor.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(
            classifier_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=1e-5
        )

        criterion = nn.BCEWithLogitsLoss() if self.multi_label else nn.CrossEntropyLoss()

        last_epoch_loss = 0.0

        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0
            
            for batch in labeled_dataloader:
                if isinstance(batch, dict):
                    x = batch["image"].to(self.device)
                    y = batch["label"].to(self.device)
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)

                batch_size = x.size(0)

                with torch.no_grad():
                    embeddings = feature_extractor(x)

                logits = classifier_head(embeddings)

                if self.multi_label:
                    loss = criterion(logits, y.float())
                else:
                    loss = criterion(logits, y.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                total_samples += batch_size

            scheduler.step()
            last_epoch_loss = total_loss / max(1, total_samples)

        return last_epoch_loss

    @torch.no_grad()
    def extract_unlabeled_embeddings_and_probs(
        self,
        feature_extractor: FeatureExtractor,
        classifier_head: ClassifierHead,
        unlabeled_dataloader: DataLoader,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature embeddings and prediction probabilities for all samples in unlabeled_dataloader.
        
        Returns:
            Tuple of (embeddings [N_u, D], probabilities [N_u, C]) as numpy arrays.
        """
        feature_extractor.to(self.device)
        feature_extractor.eval()
        classifier_head.to(self.device)
        classifier_head.eval()

        all_embeddings = []
        all_probs = []

        for batch in unlabeled_dataloader:
            if isinstance(batch, dict):
                x = batch["image"].to(self.device, non_blocking=True)
            else:
                x = batch[0].to(self.device, non_blocking=True)

            embeddings = feature_extractor(x)
            probs = classifier_head.predict_probabilities(
                embeddings,
                multi_label=self.multi_label,
                temperature=temperature,
            )

            all_embeddings.append(embeddings.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        if len(all_embeddings) == 0:
            return np.empty((0, feature_extractor.feature_dim)), np.empty((0, classifier_head.num_classes))

        embeddings_arr = np.concatenate(all_embeddings, axis=0)
        probs_arr = np.concatenate(all_probs, axis=0)

        return embeddings_arr, probs_arr

    @torch.no_grad()
    def evaluate(
        self,
        feature_extractor: FeatureExtractor,
        classifier_head: ClassifierHead,
        dataloader: DataLoader,
        num_classes: int,
    ) -> Dict[str, Any]:
        """
        Evaluate performance on test or validation dataset using standard metrics.
        """
        feature_extractor.to(self.device)
        feature_extractor.eval()
        classifier_head.to(self.device)
        classifier_head.eval()

        acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro").to(self.device)
        acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro").to(self.device)
        precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(self.device)
        recall = MulticlassRecall(num_classes=num_classes, average="macro").to(self.device)
        f1_score = MulticlassF1Score(num_classes=num_classes, average="macro").to(self.device)
        kappa = MulticlassCohenKappa(num_classes=num_classes).to(self.device)
        cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)

        all_preds = []
        all_targets = []

        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)
            else:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

            embeddings = feature_extractor(x)
            logits = classifier_head(embeddings)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_targets.append(y.long())

        if len(all_preds) == 0:
            return {}

        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        metrics = {
            "OverallAccuracy": acc_micro(preds_cat, targets_cat).item(),
            "AverageAccuracy": acc_macro(preds_cat, targets_cat).item(),
            "Precision": precision(preds_cat, targets_cat).item(),
            "Recall": recall(preds_cat, targets_cat).item(),
            "F1Score": f1_score(preds_cat, targets_cat).item(),
            "Kappa": kappa(preds_cat, targets_cat).item(),
            "ConfusionMatrix": cm_metric(preds_cat, targets_cat).cpu().numpy(),
        }

        return metrics
