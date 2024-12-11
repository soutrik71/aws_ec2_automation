import lightning as pl
import torch.nn as nn
import torch
from timm import create_model
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from loguru import logger
import os

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure Loguru to save logs to the logs/ directory
logger.add(f"{log_dir}/model.log", rotation="1 MB", level="INFO", enqueue=False)


class LitEfficientNet(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientnet_lite0",
        num_classes=10,
        lr=1e-3,
        custom_loss=None,
    ):
        """
        Initializes a CNN model from TIMM and integrates TorchMetrics.

        Args:
            model_name (str): TIMM model name (e.g., "tf_efficientnet_lite0").
            num_classes (int): Number of output classes (e.g., 0–9 for MNIST).
            lr (float): Learning rate for the optimizer.
            custom_loss (callable, optional): Custom loss function. Defaults to CrossEntropyLoss.
        """
        super().__init__()

        self.lr = lr
        self.model = create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=1,  # Set to 1 channel for grayscale input
        )
        self.loss_fn = custom_loss or nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.val_acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.test_acc = Accuracy(num_classes=num_classes, task="multiclass")
        logger.info(f"Model initialized with TIMM backbone: {model_name}")
        logger.info(f"Number of output classes: {num_classes}")

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_acc.update(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_acc.update(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_acc.update(y_hat, y)
        self.log("test_acc", self.test_acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        logger.info(f"Optimizer: Adam, Learning Rate: {self.lr}")
        logger.info("Scheduler: StepLR with step_size=1 and gamma=0.9")
        return [optimizer], [scheduler]
