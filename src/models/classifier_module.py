import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import Optional, Dict, Any

from ..backbones.registry import BACKBONE_REGISTRY
from ..learning_engine.feature_extractor import FeatureExtractor
from ..learning_engine.classifier import ClassifierHead


class ClassifierModule(LightningModule):
    """
    LightningModule for SpectralEarth Classifier with frozen backbone
    and trainable classifier head.
    """

    def __init__(
        self,
        backbone: str = "spec_resnet50",
        num_classes: int = 16,
        in_channels: int = 200,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Instantiate backbone
        if backbone in BACKBONE_REGISTRY:
            model_cls = BACKBONE_REGISTRY[backbone]
            if "in_channels" in model_cls.__init__.__code__.co_varnames:
                raw_backbone = model_cls(num_classes=num_classes, in_channels=in_channels)
            else:
                raw_backbone = model_cls(num_classes=num_classes)
        else:
            import timm
            raw_backbone = timm.create_model(backbone, num_classes=num_classes, in_chans=in_channels)

        self.feature_extractor = FeatureExtractor(backbone=raw_backbone)
        
        # Freeze backbone parameters
        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        feat_dim = self.feature_extractor.feature_dim
        self.classifier_head = ClassifierHead(in_features=feat_dim, num_classes=num_classes, hidden_dim=hidden_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.feature_extractor(x)
        logits = self.classifier_head(embeddings)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        y = batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
