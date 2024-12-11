from pathlib import Path
import torch
import rootutils
from src.model import LitEfficientNet
from loguru import logger

import os

logger_dir = "/tmp/logs"
os.makedirs(logger_dir, exist_ok=True)

# Configure loggeruru to save loggers to the loggers/ directory
logger.add(
    f"{logger_dir}/torchscript.log", rotation="1 MB", level="INFO", enqueue=False
)

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def make_jit_model(ckpt_path, input_shape=(1, 1, 28, 28)):
    # Initialize Model
    model = LitEfficientNet(model_name="tf_efficientnet_lite0", num_classes=10, lr=1e-3)

    # Check for checkpoint file
    checkpoint_file = Path(ckpt_path) / "best_model.ckpt"
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint file not found at {checkpoint_file}")
        return

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Checkpoint loaded from {checkpoint_file}")

    # Set model to evaluation mode
    model.eval()

    # Move model and input to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    example_input = torch.randn(*input_shape).to(device)
    logger.info(f"Using device: {device}")

    # Trace the model
    logger.info("Tracing model...")
    try:
        traced_model = model.to_torchscript(
            method="trace", example_inputs=example_input
        )
        logger.info("Model traced successfully!")
    except Exception as e:
        logger.error(f"Error during model tracing: {e}")
        return

    # Save the traced model
    output_dir = Path(ckpt_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "traced_model.pt"
    torch.jit.save(traced_model, output_path)
    logger.info(f"Traced model saved to: {output_path}")


if __name__ == "__main__":
    make_jit_model(ckpt_path="./checkpoints")
