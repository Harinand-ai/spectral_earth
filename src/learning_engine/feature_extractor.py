import torch
import torch.nn as nn
from typing import Optional


class FeatureExtractor(nn.Module):
    """
    Feature Extractor for Model A (Learning Engine).
    
    Wraps a selected backbone (e.g. SpectralEarth), freezes all parameters,
    and extracts flattened 1D feature embeddings for input hyperspectral images.
    """

    def __init__(self, backbone: nn.Module, feature_dim: Optional[int] = None):
        super().__init__()
        self.backbone = backbone
        
        # Freeze backbone parameters permanently
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()
        self._feature_dim = feature_dim

    def train(self, mode: bool = True):
        # Keep backbone permanently in eval mode for feature extraction
        super().train(False)
        self.backbone.eval()
        return self

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is not None:
            return self._feature_dim
        
        if hasattr(self.backbone, "num_features"):
            return self.backbone.num_features
        elif hasattr(self.backbone, "embed_dim"):
            return self.backbone.embed_dim
        elif hasattr(self.backbone, "resnet") and hasattr(self.backbone.resnet, "num_features"):
            return self.backbone.resnet.num_features
        elif hasattr(self.backbone, "vit_core") and hasattr(self.backbone.vit_core, "num_features"):
            return self.backbone.vit_core.num_features
        
        raise ValueError("Could not determine feature_dim automatically. Provide feature_dim during initialization.")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings for input tensor x.
        
        Args:
            x: Tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B, D] containing feature embeddings.
        """
        self.backbone.eval()
        
        if hasattr(self.backbone, "spectral_adapter"):
            adapted = self.backbone.spectral_adapter(x)
            if hasattr(self.backbone, "resnet"):
                sub_backbone = self.backbone.resnet
                x_feat = sub_backbone.layer1(adapted)
                x_feat = sub_backbone.layer2(x_feat)
                x_feat = sub_backbone.layer3(x_feat)
                x_feat = sub_backbone.layer4(x_feat)
                feats = sub_backbone.global_pool(x_feat).flatten(1)
                return feats
            elif hasattr(self.backbone, "vit_core"):
                feats = self.backbone.vit_core.forward_features(adapted)
                if hasattr(self.backbone.vit_core, "forward_head"):
                    feats = self.backbone.vit_core.forward_head(feats, pre_logits=True)
                return feats
            else:
                feats = adapted
        elif hasattr(self.backbone, "get_backbone"):
            sub_backbone = self.backbone.get_backbone()
            if hasattr(sub_backbone, "forward_features"):
                feats = sub_backbone.forward_features(x)
            elif hasattr(sub_backbone, "return_features"):
                orig_setting = getattr(sub_backbone, "return_features", False)
                sub_backbone.return_features = True
                feats = sub_backbone(x)
                sub_backbone.return_features = orig_setting
            else:
                feats = sub_backbone(x)
        elif hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if hasattr(self.backbone, "forward_head"):
                feats = self.backbone.forward_head(feats, pre_logits=True)
        elif hasattr(self.backbone, "return_features"):
            orig_setting = getattr(self.backbone, "return_features", False)
            self.backbone.return_features = True
            feats = self.backbone(x)
            self.backbone.return_features = orig_setting
        else:
            feats = self.backbone(x)
            
        # Standardize output feature tensor to 2D [B, D]
        if feats.ndim == 4:
            feats = torch.mean(feats, dim=(2, 3))
        elif feats.ndim == 3:
            feats = torch.mean(feats, dim=1)
            
        return feats

