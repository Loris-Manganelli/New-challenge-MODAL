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

        mixup_dataset = MultiLabelDataset(self.mixup_dir, self.train_transform)
        return DataLoader(
            ConcatDataset([self.dataset, mixup_dataset]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )



class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of all image filenames
        self.filenames = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path)
        
        image = self.transform(image)

        
        # Extract the class labels from the filename
        labels = filename.split('_')[:2]  # Get the first two parts of the filename
        labels = [(label, 0.5) for label in labels]  # Assign a weight of 0.5 to each class


        num_classes = len(set(label[0] for label in labels))
        label_tensor = torch.zeros(num_classes)
        for label in labels:
            label_idx = int(label[0].split('class')[1]) - 1  # Assuming the class labels are in the format 'classX'
            label_tensor[label_idx] = label[1]

        # Make sure the image and label tensors have the correct shape and data type
        image = image.float()  # Convert the image to float32
        label_tensor = label_tensor.unsqueeze(0)  # Add a batch dimension to the label tensor

        return image, label_tensor