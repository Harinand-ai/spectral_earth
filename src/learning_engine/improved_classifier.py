import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ImprovedClassifierHead(nn.Module):
    """
    Improved Classifier Head for Frozen SpectralEarth Foundation Embeddings.
    
    Architectural enhancements based on diagnostic findings:
    1. Input L2-normalization projection (unit hypersphere mapping in R^2048)
    2. Higher capacity hidden representation (512 units)
    3. Batch normalization / LayerNorm for stable feature scaling across classes
    4. Lightweight dropout (0.10) for regularized generalization
    5. Fully compatible with Cosine Annealing learning rate schedule
    """

    def __init__(
        self,
        in_features: int = 2048,
        num_classes: int = 16,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if norm_type.lower() == "batchnorm":
            norm1 = nn.BatchNorm1d(hidden_dim)
            norm2 = nn.BatchNorm1d(hidden_dim // 2)
        else:
            norm1 = nn.LayerNorm(hidden_dim)
            norm2 = nn.LayerNorm(hidden_dim // 2)

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            norm1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            norm2,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._reset_weights()

    def _reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
        if normalize_input:
            x = x / torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=1e-8)
        
        # Guard against single-sample batch causing BatchNorm crash in training mode
        if self.training and x.dim() > 1 and x.size(0) == 1:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)

