import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    Trainable Classifier Head for Model A (Learning Engine).
    
    Receives feature embeddings of dimension `in_features` from the frozen backbone
    and predicts class logits and probabilities for `num_classes`.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(in_features)

        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing raw logits.
        
        Args:
            features: Tensor of shape [B, D]
            
        Returns:
            Logits tensor of shape [B, C]
        """
        features = self.input_norm(features)
        return self.head(features)

    def predict_probabilities(
        self,
        features: torch.Tensor,
        multi_label: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute normalized class probabilities.
        
        Args:
            features: Tensor of shape [B, D]
            multi_label: If True, uses Sigmoid; otherwise Softmax.
            temperature: Temperature scaling factor for probability calibration.
            
        Returns:
            Probability tensor of shape [B, C]
        """
        logits = self.forward(features)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        if multi_label:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
