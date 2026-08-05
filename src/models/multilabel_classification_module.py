import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules import loss
import timm
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
)
from lightning.pytorch import LightningModule

from ..backbones.registry import BACKBONE_REGISTRY

# import the list type
from typing import List
class MultiLabelClassificationModule(LightningModule):
    """LightningModule for multi-label hyperspectral image classification.
    
    This module implements training, validation, and testing pipelines for
    multi-label classification using hyperspectral imagery. It supports various
    backbone architectures, fine-tuning strategies, and metric tracking.
    """

    def __init__(
        self,
        backbone: str, 
        num_classes: int,
        in_channels: int = 202,
        img_size: int = 128,
        pretrained_weights: Optional[str] = None,
        token_patch_size: Optional[int] = None,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        finetune_first_n_layers: int = 0,
        loss: str = "bce",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 100,
        eta_min: float = 0.0,
    ) -> None:
        """Initialize the Lightning multi-label classification module.

        Args:
            backbone: Name of backbone model (from registry or timm)
            num_classes: Number of target classes for classification
            in_channels: Number of input spectral bands (default: 202 for EnMAP)
            img_size: Input image size (default: 128)
            pretrained_weights: Path to pretrained weights file (optional)
            token_patch_size: Patch size for ViT models (optional)
            freeze_backbone: If True, freeze backbone for linear probing
            finetune_adapter: If True, only finetune spectral adapter
            finetune_first_n_layers: Number of backbone layers to unfreeze
            loss: Loss function type (currently only "bce" supported)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            t_max: Maximum iterations for cosine scheduler
            eta_min: Minimum learning rate for scheduler
        """
        super().__init__()

        self.backbone_name = backbone  
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights
        self.token_patch_size = token_patch_size
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.finetune_first_n_layers = finetune_first_n_layers
        self.loss_name = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min
        self.in_channels = in_channels
        self.img_size = img_size


        # Configure the model, loss, and metrics
        self._config_task()
        self._configure_metrics()

    def _config_task(self) -> None:
        """Configure the model and loss function."""
        # Initialize the model
        self._initialize_model()

        # Set the loss function
        if self.loss_name == "bce":
          self.loss = nn.BCEWithLogitsLoss()

        elif self.loss_name == "ce":
            self.loss = nn.CrossEntropyLoss()

        else:
            raise ValueError(f"Loss type '{self.loss_name}' is not valid.")

        # Freeze backbone if specified
        if self.freeze_backbone:
            self._freeze_backbone()

        # Unfreeze adapter layers if finetuning is enabled
        if self.finetune_adapter:
            self._unfreeze_adapter_layers()
            
        if self.finetune_first_n_layers > 0:
            self._unfreeze_first_n_layers()

    def _initialize_model(self) -> None:
        """Initialize the classification model using the provided backbone name."""
        if self.backbone_name in BACKBONE_REGISTRY:
            model_cls = BACKBONE_REGISTRY[self.backbone_name]
            if "vit" in self.backbone_name:
                self.model = model_cls(
                    num_classes=self.num_classes,
                    token_patch_size=self.token_patch_size,
                )
            else:
                # Check if in_channels is part of the model's constructor
                if "in_channels" in model_cls.__init__.__code__.co_varnames:
                    self.model = model_cls(
                        num_classes=self.num_classes,
                        in_channels=self.in_channels,
                    )
                else:
                    self.model = model_cls(num_classes=self.num_classes)
        else:
            # Use a timm model if the backbone is not in our registry
            if "vit" in self.backbone_name:
                self.model = timm.create_model(
                    self.backbone_name,
                    num_classes=self.num_classes,
                    pretrained=False,
                    in_chans=self.in_channels,
                    img_size=self.img_size,
                    patch_size=self.token_patch_size,
                )
            else:
                self.model = timm.create_model(
                    self.backbone_name,
                    num_classes=self.num_classes,
                    pretrained=False,
                    in_chans=self.in_channels,
                )

        self._load_pretrained_weights()

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights into the model if a path is provided."""
        if self.pretrained_weights:
            if not os.path.exists(self.pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights not found at {self.pretrained_weights}")
            state_dict = torch.load(self.pretrained_weights, map_location="cpu")
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded:", msg)

    def _freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head if present
        if hasattr(self.model, 'get_classifier'):
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            print("Warning: Model does not have a recognizable classifier head to unfreeze.")

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreeze the adapter layers."""
        if hasattr(self.model, 'spectral_adapter'):
            for param in self.model.spectral_adapter.parameters():
                param.requires_grad = True
        else:
            raise AttributeError("The backbone does not have 'spectral_adapter' attributes.")

    def _unfreeze_first_n_layers(self) -> None:
        """
        Unfreezes the first n layers of the backbone while ignoring the spectral adapter.
        For ResNet-based models, this will unfreeze resnet.layer1, layer2, etc.
        For ViT-based models, this will unfreeze the first n transformer blocks in vit_core.blocks.
        """
        n = self.finetune_first_n_layers
        if n <= 0:
            return

        # For a ResNet-based backbone (ignoring the spectral adapter)
        if hasattr(self.model, "resnet"):
            resnet = self.model.resnet
            layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            for i, layer in enumerate(layers):
                if i < n:
                    for param in layer.parameters():
                        param.requires_grad = True

        # For a ViT-based backbone (ignoring the spectral adapter)
        elif hasattr(self.model, "vit_core"):
            vit = self.model.vit_core
            for i, block in enumerate(vit.blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = True
                        
    def _configure_metrics(self) -> None:
    

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                ),
                "AverageAccuracy": MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "Precision": MulticlassPrecision(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "Recall": MulticlassRecall(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "F1Score": MulticlassF1Score(
                    num_classes=self.num_classes,
                    average="macro",
                ),
                "Kappa": MulticlassCohenKappa(
                    num_classes=self.num_classes,
                ),
            },
            prefix="train_",
        )

        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=self.num_classes
        )

        self.best_val_loss = float("inf")
        self.best_epoch = None

        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
         self.max_val_metrics["max_" + key] = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Training loss tensor.
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        if self.loss_name == "ce":
            loss = self.loss(y_hat, y)

            preds = torch.argmax(y_hat, dim=1)
            self.train_metrics(preds, y)

        else:
            y_hat = torch.sigmoid(y_hat)
            loss = self.loss(y_hat, y.float())
            self.train_metrics(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss and metrics.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        if self.loss_name == "ce":
            loss = self.loss(y_hat, y)

            preds = torch.argmax(y_hat, dim=1)
            self.val_metrics(preds, y)

        else:
            y_hat = torch.sigmoid(y_hat)
            loss = self.loss(y_hat, y.float())
            self.val_metrics(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    def test_step(self, batch: Dict[str, Tensor], batch_idx: int):

        x = batch["image"]
        y = batch["label"]

        y_hat = self(x)

        if self.loss_name == "ce":

            loss = self.loss(y_hat, y)

            preds = torch.argmax(y_hat, dim=1)

            self.test_metrics(preds, y)

            self.confusion_matrix(preds, y)

        else:

            y_hat = torch.sigmoid(y_hat)

            loss = self.loss(y_hat, y.float())

            self.test_metrics(y_hat, y)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            Predicted sigmoid probabilities tensor.
        """
        x = batch["image"]
        y_hat = torch.sigmoid(self(x))
        return y_hat

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()

        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        self.val_metrics.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()

        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)

        cm = self.confusion_matrix.compute()

        print("\n==============================")
        print("SpectralEarth Evaluation")
        print("==============================")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        self.test_metrics.reset()
        self.confusion_matrix.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }