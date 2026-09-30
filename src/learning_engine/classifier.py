import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    Trainable Classifier Head for Model A (Learning Engine).
    
    Supports:
    - Linear projection head
    - 2-layer & 3-layer MLP heads
    - Residual MLP head
    - Configurable activations (ReLU, GELU)
    - LayerNorm / BatchNorm / None
    - Kaiming / Xavier weight initialization
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
        head_type: str = "linear",
        activation: str = "relu",
        norm_layer: str = "layernorm",
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.head_type = head_type.lower()

        # Input representation normalization
        if norm_layer.lower() == "layernorm":
            self.input_norm = nn.LayerNorm(in_features)
        elif norm_layer.lower() == "batchnorm":
            self.input_norm = nn.BatchNorm1d(in_features)
        else:
            self.input_norm = nn.Identity()

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        if hidden_dim <= 0 or self.head_type == "linear":
            self.head = nn.Linear(in_features, num_classes)
        elif self.head_type == "mlp_2layer":
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim) if norm_layer == "layernorm" else nn.Identity(),
                act_cls(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        elif self.head_type == "mlp_3layer":
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim) if norm_layer == "layernorm" else nn.Identity(),
                act_cls(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2) if norm_layer == "layernorm" else nn.Identity(),
                act_cls(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
        elif self.head_type == "residual_mlp":
            self.dense1 = nn.Linear(in_features, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim) if norm_layer == "layernorm" else nn.Identity()
            self.act1 = act_cls()
            self.drop1 = nn.Dropout(p=dropout)
            self.dense2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim) if norm_layer == "layernorm" else nn.Identity()
            self.act2 = act_cls()
            self.proj_out = nn.Linear(hidden_dim, num_classes)
            self.skip_proj = nn.Linear(in_features, hidden_dim) if in_features != hidden_dim else nn.Identity()
            self.head = None  # Handled in custom forward
        else:
            self.head = nn.Linear(in_features, num_classes)

        self._reset_weights()

    def _reset_weights(self):
        """Initialize linear weights using Kaiming normal / Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing raw logits.
        """
        features = self.input_norm(features)
        if self.head_type == "residual_mlp" and hasattr(self, "dense1"):
            residual = self.skip_proj(features)
            out = self.drop1(self.act1(self.norm1(self.dense1(features))))
            out = self.act2(self.norm2(self.dense2(out))) + residual
            return self.proj_out(out)
        return self.head(features)

    def predict_probabilities(
        self,
        features: torch.Tensor,
        multi_label: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute normalized class probabilities with optional temperature scaling.
        """
        logits = self.forward(features)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
