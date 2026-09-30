import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, Dataset

class CachedEmbeddingDataset(Dataset):
    """Dataset wrapper for pre-extracted feature embeddings and targets."""
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class FeatureCache:
    """
    Feature Caching Manager for Frozen SpectralEarth Backbone.
    
    Extracts 2048-dimensional feature embeddings once per dataset and provides
    fast batch lookups for training and active learning sample selection.
    """
    def __init__(self, feature_extractor, device: Optional[torch.device] = None):
        self.feature_extractor = feature_extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def precompute_dataset(self, dataset_name: str, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and cache embeddings and labels for an entire dataloader."""
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        all_embeddings = []
        all_labels = []

        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch["image"].to(self.device, non_blocking=True)
                y = batch["label"].to(self.device, non_blocking=True)
            else:
                x, y = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)

            feats = self.feature_extractor(x)
            all_embeddings.append(feats.cpu())
            all_labels.append(y.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        self.cache[dataset_name] = (embeddings, labels)
        return embeddings, labels

    def get_cached_dataloader(self, dataset_name: str, indices: Optional[np.ndarray] = None, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
        """Get DataLoader serving cached embeddings for specified indices."""
        if dataset_name not in self.cache:
            raise KeyError(f"Dataset '{dataset_name}' is not precomputed in FeatureCache.")

        embeddings, labels = self.cache[dataset_name]

        if indices is not None:
            indices_t = torch.as_tensor(indices, dtype=torch.long)
            embeddings = embeddings[indices_t]
            labels = labels[indices_t]

        cached_ds = CachedEmbeddingDataset(embeddings, labels)
        return DataLoader(cached_ds, batch_size=batch_size, shuffle=shuffle)
