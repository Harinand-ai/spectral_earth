from torchgeo.datasets.geo import NonGeoDataset
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
import os
from typing import Optional
import matplotlib.pyplot as plt


class IndianPinesDataset(NonGeoDataset):
    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "datasets/IndianPines",
        split: str = "train",
        patch_size: int = 15,
    ) -> None:

        if split not in self.valid_splits:
            raise ValueError(f"Split must be one of {self.valid_splits}")

        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.margin = patch_size // 2

        image_path = os.path.join(root, "Indian_pines_corrected.mat")
        label_path = os.path.join(root, "Indian_pines_gt.mat")

        image = loadmat(image_path)["indian_pines_corrected"]
        labels = loadmat(label_path)["indian_pines_gt"]

        image = image.astype(np.float32)

        mean = image.mean(axis=(0, 1))
        std = image.std(axis=(0, 1)) + 1e-8
        image = (image - mean) / std

        self.image = np.pad(
            image,
            ((self.margin, self.margin),
             (self.margin, self.margin),
             (0, 0)),
            mode="reflect",
        )

        self.labels = labels

        coords = np.argwhere(labels > 0)

        np.random.seed(42)
        np.random.shuffle(coords)

        n = len(coords)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if split == "train":
            self.samples = coords[:train_end]
        elif split == "val":
            self.samples = coords[train_end:val_end]
        else:
            self.samples = coords[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        r, c = self.samples[index]

        r_pad = r + self.margin
        c_pad = c + self.margin

        patch = self.image[
            r_pad - self.margin:r_pad + self.margin + 1,
            c_pad - self.margin:c_pad + self.margin + 1,
            :
        ]

        patch = torch.tensor(patch).permute(2, 0, 1).float()

        label = int(self.labels[r, c]) - 1

        return {
            "image": patch,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ):

        fig, ax = plt.subplots(figsize=(5, 5))

        rgb = sample["image"][[29, 19, 9]].permute(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        ax.imshow(rgb)

        if show_titles:
            ax.set_title(f"Class {sample['label'].item()+1}")

        if suptitle:
            fig.suptitle(suptitle)

        return fig