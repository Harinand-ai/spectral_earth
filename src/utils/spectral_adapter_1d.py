import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralResampler1D(nn.Module):
    """
    Isolated deterministic 1D spectral resampling module.
    Resamples hyperspectral input patch tensors [B, C_in, H, W] to [B, C_out, H, W]
    along the spectral (channel) dimension using 1D linear interpolation.
    
    This module has 0 trainable parameters and leaves backbone weights untouched.
    """
    def __init__(self, target_bands: int):
        super().__init__()
        self.target_bands = target_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C_in, H, W]
            
        Returns
        -------
        torch.Tensor
            Resampled tensor of shape [B, target_bands, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C_in, H, W], got shape {x.shape}")
        
        b, c_in, h, w = x.shape
        if c_in == self.target_bands:
            return x

        # Reshape to [B * H * W, 1, C_in] for 1D interpolation along spectral axis
        x_perm = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, 1, c_in)
        
        # Linear interpolation along spectral dimension C_in -> C_out
        x_resampled = F.interpolate(x_perm, size=self.target_bands, mode="linear", align_corners=True)
        
        # Reshape back to [B, C_out, H, W]
        x_out = x_resampled.view(b, h, w, self.target_bands).permute(0, 3, 1, 2).contiguous()
        return x_out
