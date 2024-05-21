from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch
import torchvision.transforms
from torchvision.transforms import v2


class DataModule:
    def __init__(
        self,
        train_dataset_path,
        real_images_val_path,
        train_transform1,
        train_transform2,
        train_transform3,
        train_transform4,
        train_transform5,
        val_transform,
        batch_size,
        num_workers,
    ):
        self.dataset1 = ImageFolder(train_dataset_path, transform=train_transform1)
        self.dataset2 = ImageFolder(train_dataset_path, transform=train_transform2)
        self.dataset3 = ImageFolder(train_dataset_path, transform=train_transform3)
        self.dataset4 = ImageFolder(train_dataset_path, transform=train_transform4)
        self.dataset5 = ImageFolder(train_dataset_path, transform=train_transform5)

        self.train_dataset1, self.val_dataset1 = torch.utils.data.random_split(
            self.dataset1,
            [
                int(0.8 * len(self.dataset1)),
                len(self.dataset1) - int(0.8 * len(self.dataset1)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        self.train_dataset2, self.val_dataset2 = torch.utils.data.random_split(
            self.dataset2,
            [
                int(0.8 * len(self.dataset2)),
                len(self.dataset2) - int(0.8 * len(self.dataset2)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        self.train_dataset3, self.val_dataset3 = torch.utils.data.random_split(
            self.dataset3,
            [
                int(0.8 * len(self.dataset3)),
                len(self.dataset3) - int(0.8 * len(self.dataset3)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        self.train_dataset4, self.val_dataset4 = torch.utils.data.random_split(
            self.dataset4,
            [
                int(0.8 * len(self.dataset4)),
                len(self.dataset4) - int(0.8 * len(self.dataset4)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        self.train_dataset5, self.val_dataset5 = torch.utils.data.random_split(
            self.dataset5,
            [
                int(0.8 * len(self.dataset5)),
                len(self.dataset5) - int(0.8 * len(self.dataset5)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )



        self.combined_training_dataset = ConcatDataset([self.train_dataset1, self.train_dataset2, self.train_dataset3, self.train_dataset4, self.train_dataset5])
        self.combined_val_dataset = ConcatDataset([self.val_dataset1, self.val_dataset2, self.val_dataset3, self.val_dataset4, self.val_dataset5])


        self.combined_val_dataset.transform = val_transform
        self.real_images_val_dataset = ImageFolder(
            real_images_val_path, transform=val_transform
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.idx_to_class = {v: k for k, v in self.dataset1.class_to_idx.items()}

    def train_dataloader(self):
        return DataLoader(
            self.combined_training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return {
            "synthetic_val": DataLoader(
                self.combined_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            "real_val": DataLoader(
                self.real_images_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        }
