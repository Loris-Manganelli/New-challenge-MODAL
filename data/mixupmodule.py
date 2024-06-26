from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch
import torchvision.transforms
from torchvision.transforms import v2
import numpy as np
import os
from PIL import Image


class DataModule:
    def __init__(
        self,
        train_dataset_path,
        real_images_val_path,
        train_transform,
        val_transform,
        batch_size,
        num_workers,
        mixup_dir,
    ):
        self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.train_dataset_path = train_dataset_path
        self.train_transform = train_transform
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )

        self.val_dataset.transform = val_transform
        self.real_images_val_dataset = ImageFolder(
            real_images_val_path, transform=val_transform
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}
        self.mixup_dir = mixup_dir


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return {
            "synthetic_val": DataLoader(
                self.val_dataset,
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
    
    def concat_dataloader(self):

        mixup_dataset = MultiLabelDataset(self.mixup_dir, self.train_dataset_path, self.train_transform)
        return DataLoader(
            ConcatDataset([self.dataset, mixup_dataset]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )



class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train_dir, transform=None,):
        self.root_dir = root_dir
        self.transform = transform
        self.train_dir = train_dir

        # Get the list of all image filenames
        self.filenames = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path)
        
        image = self.transform(image)
        classes = os.listdir(self.train_dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        # Extract the class labels from the filename
        labels = filename.split('_')[:2]  # Get the first two parts of the filename

        target = torch.zeros(len(classes))
        target[[class_to_idx[labels[0]], class_to_idx[labels[1]]]] = 0.5

        #labels = [(label, 0.5) for label in labels]  # Assign a weight of 0.5 to each class



        return image, target
    
