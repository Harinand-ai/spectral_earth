import torch
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
from typing import Optional, List, Union, Tuple


class ActiveLearningDatasetManager:
    """
    Dataset Manager for Model A (Learning Engine).
    
    Splits training data into Labeled Pool (L) and Unlabeled Pool (U).
    Tracks pool state and updates pools after each Active Learning query round.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        initial_labeled_ratio: float = 0.05,
        initial_samples_per_class: Optional[int] = None,
        seed: int = 42,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.seed = seed

        total_samples = len(train_dataset)
        all_indices = np.arange(total_samples)

        rng = np.random.RandomState(seed)

        if initial_samples_per_class is not None and initial_samples_per_class > 0:
            # Class-balanced initial seed
            labels = self._extract_labels(train_dataset)
            unique_classes = np.unique(labels)
            initial_labeled = []
            
            for c in unique_classes:
                c_indices = np.where(labels == c)[0]
                rng.shuffle(c_indices)
                n_take = min(len(c_indices), initial_samples_per_class)
                initial_labeled.extend(c_indices[:n_take])
                
            self.labeled_indices = np.array(initial_labeled, dtype=np.int64)
        else:
            # Random initial ratio seed
            num_initial = max(1, int(total_samples * initial_labeled_ratio))
            shuffled = all_indices.copy()
            rng.shuffle(shuffled)
            self.labeled_indices = shuffled[:num_initial].astype(np.int64)

        # Unlabeled pool is all non-labeled indices
        mask = np.ones(total_samples, dtype=bool)
        mask[self.labeled_indices] = False
        self.unlabeled_indices = all_indices[mask].astype(np.int64)

    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        if hasattr(dataset, "samples") and hasattr(dataset, "labels"):
            coords = dataset.samples
            return np.array([int(dataset.labels[r, c]) - 1 for r, c in coords])
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict):
                lbl = sample["label"]
            elif isinstance(sample, (tuple, list)):
                lbl = sample[1]
            else:
                lbl = sample
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            labels.append(lbl)
        return np.array(labels)

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_indices)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_indices)

    def update_pools(self, selected_unlabeled_indices: np.ndarray) -> np.ndarray:
        """
        Move selected samples from Unlabeled Pool to Labeled Pool.
        
        Args:
            selected_unlabeled_indices: Indices relative to the current unlabeled pool.
            
        Returns:
            Global dataset indices of the newly added samples.
        """
        selected_unlabeled_indices = np.asarray(selected_unlabeled_indices, dtype=np.int64)
        global_indices = self.unlabeled_indices[selected_unlabeled_indices]

        # Add to labeled pool
        self.labeled_indices = np.concatenate([self.labeled_indices, global_indices])

        # Remove from unlabeled pool
        mask = np.ones(len(self.unlabeled_indices), dtype=bool)
        mask[selected_unlabeled_indices] = False
        self.unlabeled_indices = self.unlabeled_indices[mask]

        return global_indices

    def get_labeled_dataset(self) -> Dataset:
        return Subset(self.train_dataset, self.labeled_indices)

    def get_unlabeled_dataset(self) -> Dataset:
        return Subset(self.train_dataset, self.unlabeled_indices)

    def get_labeled_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_labeled_dataset(),
            batch_size=min(batch_size, len(self.labeled_indices)),
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def get_unlabeled_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        return DataLoader(
            self.get_unlabeled_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def get_val_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **kwargs,
        )

    def get_test_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            **kwargs,
        )
