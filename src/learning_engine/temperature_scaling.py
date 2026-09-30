import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for Logit Calibration.
    
    Fits a single scalar parameter T > 0 on validation set logits via NLL loss minimization.
    Does NOT modify the classifier or backbone parameters.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float:
        """
        Fit temperature parameter T using validation logits and labels ONLY.
        """
        val_logits = val_logits.clone().detach()
        val_labels = val_labels.clone().detach().long()

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        
        # Clamp temperature to positive range [0.05, 10.0]
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)

        return float(self.temperature.item())

    @torch.no_grad()
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities from logits."""
        scaled_logits = self.forward(logits)
        return torch.softmax(scaled_logits, dim=1)
