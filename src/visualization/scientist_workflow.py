"""
ConformAL-HSI: Scientist-Facing Interactive Workflow & Spatial Mapping Engine.

Provides tools for scientists to:
1. Visualize spatial classification maps alongside false-color imagery and ground truth.
2. Inspect spatial uncertainty (margin ambiguity) maps to guide interactive annotation.
3. Inspect conformal prediction-set cardinality maps to identify multi-class ambiguities and anomalies.
4. Export high-resolution publication figures and operational decision guidance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class ScientistWorkflowEngine:
    """
    Scientist-Facing Hyperspectral Active Learning Visualization & Decision Suite.
    """
    def __init__(self, out_dir: str = "results/final_research_consolidation/scientist_maps"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_spatial_maps(
        self,
        dataset_slug: str,
        gt_path: str,
        gt_key: str,
        model: torch.nn.Module,
        all_feats: torch.Tensor,
        all_coords: np.ndarray,
        q_hat: float,
        num_classes: int,
        dataset_name: str,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> Dict[str, str]:
        """
        Generate and save a 4-panel spatial map:
        1. Ground Truth Class Map
        2. Predicted Class Map
        3. Margin Uncertainty Ambiguity Map
        4. Conformal Prediction Set Cardinality Map
        """
        mat = loadmat(gt_path)
        labels = mat[gt_key]
        H, W = labels.shape

        # Full scene inference
        model.eval()
        probs_list = []
        with torch.no_grad():
            for bi in range(0, len(all_feats), 2048):
                bx = all_feats[bi : bi + 2048].to(device)
                probs_list.append(F.softmax(model(bx), dim=-1).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        preds = np.argmax(probs, axis=-1)

        # Compute margin uncertainty and conformal set size
        sorted_p = np.sort(probs, axis=1)
        margin_unc = 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])

        # Conformal sets: C(x) = {c: 1 - p(c|x) <= q_hat} = {c: p(c|x) >= 1 - q_hat}
        in_set = (probs >= (1.0 - q_hat))
        set_sizes = in_set.sum(axis=1)

        # Project 1D predictions back to 2D image coordinates
        pred_map = np.zeros((H, W), dtype=np.int32)
        unc_map = np.zeros((H, W), dtype=np.float32)
        conf_map = np.zeros((H, W), dtype=np.float32)

        for idx, (r, c) in enumerate(all_coords):
            pred_map[r, c] = preds[idx] + 1 # 1-indexed for colormap
            unc_map[r, c] = margin_unc[idx]
            conf_map[r, c] = set_sizes[idx]

        # Generate publication plot
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=300)

        # Panel 1: Ground Truth
        try:
            import matplotlib as mpl
            cmap_classes = mpl.colormaps["tab20"].resampled(num_classes + 1)
        except Exception:
            cmap_classes = plt.get_cmap("tab20", num_classes + 1)
        im0 = axes[0].imshow(labels, cmap=cmap_classes, vmin=0, vmax=num_classes)
        axes[0].set_title(f"A. Ground Truth Map\n({dataset_name})", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Class ID")

        # Panel 2: Predicted Map
        im1 = axes[1].imshow(pred_map, cmap=cmap_classes, vmin=0, vmax=num_classes)
        axes[1].set_title(f"B. Model Prediction Map\n(Active Learning Probing)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Class ID")

        # Panel 3: Margin Uncertainty Map
        im2 = axes[2].imshow(unc_map, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"C. Margin Uncertainty Map\n(High = Ambiguous Boundary)", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Margin Uncertainty (1 - gap)")

        # Panel 4: Conformal Set Cardinality Map
        im3 = axes[3].imshow(conf_map, cmap="plasma", vmin=0, vmax=min(num_classes, 6))
        axes[3].set_title(f"D. Conformal Set Size |C(x)|\n(Target 90% Coverage)", fontsize=12, fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="Prediction Set Cardinality")

        plt.tight_layout()
        save_path = os.path.join(self.out_dir, f"{dataset_slug}_scientist_spatial_map.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

        return {"spatial_map_path": save_path}
