import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
from src.dataloader import MNISTDataModule
from src.model import LitEfficientNet
from loguru import logger
import os

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure Loguru for logging
logger.add("logs/training.log", rotation="1 MB", level="INFO")


def main():
    """
    Main training loop for the model with advanced configuration (CPU training).
    """
    # Data Module
    logger.info("Setting up data module...")
    data_module = MNISTDataModule(batch_size=64)

    # Model
    logger.info("Setting up model...")
    model = LitEfficientNet(model_name="tf_efficientnet_lite0", num_classes=10, lr=1e-3)
    logger.info(model)

    # Callbacks
    logger.info("Setting up callbacks...")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="best_model",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
        verbose=True,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=5,  # Extended patience for advanced models
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")  # Log learning rate
    rich_progress = RichProgressBar()
    model_summary = ModelSummary(
        max_depth=1
    )  # Show only the first level of model layers

    # Loggers
    logger.info("Setting up loggers...")
    csv_logger = CSVLogger("logs/", name="mnist_csv")
    tb_logger = TensorBoardLogger("logs/", name="mnist_tb")

    # Trainer Configuration for CPU
    logger.info("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
            rich_progress,
            model_summary,
        ],
        logger=[csv_logger, tb_logger],
        deterministic=True,
        accelerator="auto",
        devices="auto",
    )

    # Train the model
    logger.info("Training the model...")
    trainer.fit(model, datamodule=data_module)

    # Test the model
    logger.info("Testing the model...")
    data_module.setup(stage="test")
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
