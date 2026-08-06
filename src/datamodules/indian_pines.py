from torchgeo.datamodules import NonGeoDataModule

from src.datasets.indian_pines import IndianPinesDataset


class IndianPinesDataModule(NonGeoDataModule):
    def __init__(
        self,
        root: str = "datasets/IndianPines",
        batch_size: int = 64,
        patch_size: int = 15,
        num_workers: int = 4,
        **kwargs,
    ) -> None:

        super().__init__(
            IndianPinesDataset,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.root = root
        self.patch_size = patch_size
        self.num_bands = 200
        self.num_classes = 16

    def setup(self, stage=None):

        if stage in (None, "fit"):
            self.train_dataset = IndianPinesDataset(
                root=self.root,
                split="train",
                patch_size=self.patch_size,
            )

            self.val_dataset = IndianPinesDataset(
                root=self.root,
                split="val",
                patch_size=self.patch_size,
            )

        if stage in (None, "test"):
            self.test_dataset = IndianPinesDataset(
                root=self.root,
                split="test",
                patch_size=self.patch_size,
            )