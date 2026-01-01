import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple
from src.config import ProjectConfig

class MNISTDataModule:
    """
    Encapsulates logic for downloading and preparing the MNIST dataset.
    """
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        # Standard normalization for MNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns Training and Testing DataLoaders.
        """
        self.config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_dataset = datasets.MNIST(
            root=str(self.config.RAW_DATA_DIR),
            train=True,
            download=True,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root=str(self.config.RAW_DATA_DIR),
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )

        return train_loader, test_loader