from loguru import logger
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import lightning as pl
from typing import Optional
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
import os

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure Loguru to save logs to the logs/ directory
logger.add(f"{log_dir}/dataloader.log", rotation="1 MB", level="INFO", enqueue=False)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        data_dir: str = "./data",
        num_workers: int = int(cpu_count()),
        train_subset_fraction: float = 0.25,  # Fraction of training data to use
    ):
        """
        Initializes the MNIST Data Module with configurations for dataloaders.

        Args:
            batch_size (int): Batch size for training, validation, and testing.
            data_dir (str): Directory to download and store the dataset.
            num_workers (int): Number of workers for data loading.
            train_subset_fraction (float): Fraction of training data to use (0.0 < fraction <= 1.0).
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_subset_fraction = train_subset_fraction
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        logger.info(f"MNIST DataModule initialized with batch size {self.batch_size}")

    def prepare_data(self):
        """
        Downloads the MNIST dataset if not already downloaded.
        """
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)
        logger.info("MNIST dataset downloaded.")

    def setup(self, stage: Optional[str] = None):
        """
        Set up the dataset for different stages.

        Args:
            stage (str, optional): One of "fit", "validate", "test", or "predict".
        """
        logger.info(f"Setting up data for stage: {stage}")
        if stage == "fit" or stage is None:
            full_train_dataset = datasets.MNIST(
                root=self.data_dir, train=True, transform=self.transform
            )
            train_indices, _ = train_test_split(
                range(len(full_train_dataset)),
                train_size=self.train_subset_fraction,
                random_state=42,
            )
            self.mnist_train = Subset(full_train_dataset, train_indices)

            self.mnist_val = datasets.MNIST(
                root=self.data_dir, train=False, transform=self.transform
            )
            logger.info(f"Loaded training subset: {len(self.mnist_train)} samples.")
            logger.info(f"Loaded validation data: {len(self.mnist_val)} samples.")
        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(
                root=self.data_dir, train=False, transform=self.transform
            )
            logger.info(f"Loaded test data: {len(self.mnist_test)} samples.")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: Training data loader.
        """
        logger.info("Creating training DataLoader...")
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: Validation data loader.
        """
        logger.info("Creating validation DataLoader...")
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: Test data loader.
        """
        logger.info("Creating test DataLoader...")
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
