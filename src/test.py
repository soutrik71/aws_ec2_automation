import torch
from loguru import logger
from src.model import LitEfficientNet
from src.dataloader import MNISTDataModule
from torchmetrics.classification import Accuracy
from pathlib import Path
from src.utils.aws_s3_services import S3Handler

# Configure Loguru to save logs to the logs/ directory
logger.add("logs/test.log", rotation="1 MB", level="INFO", enqueue=False)


def infer(checkpoint_path, image):
    """
    Perform inference on a single image using the model checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        image (torch.Tensor): Image tensor to predict (shape: [1, 28, 28] for MNIST).

    Returns:
        int: Predicted class (0-9).
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path} for inference...")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference will run on device: {device}")

    # Load the model
    model = LitEfficientNet.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # Perform inference
    with torch.no_grad():
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
        image = image.to(device)  # Ensure the image is on the same device as the model
        prediction = model(image)
        predicted_class = torch.argmax(prediction, dim=1).item()

    logger.info(f"Predicted class: {predicted_class}")
    return predicted_class


def test_model(checkpoint_path):
    """
    Test the model using the test dataset and log metrics.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        float: Final test accuracy.
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path} for testing...")
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing will run on device: {device}")

    # Load the model
    model = LitEfficientNet.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # Set up data module and load test data
    data_module = MNISTDataModule()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    # Initialize accuracy metric
    test_acc = Accuracy(num_classes=10, task="multiclass").to(device)

    # Evaluate model on test data
    logger.info("Evaluating on test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(
                device
            )  # Move data to the same device
            outputs = model(images)
            test_acc.update(outputs, labels)

    accuracy = test_acc.compute().item()
    logger.info(f"Final Test Accuracy (TorchMetrics): {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":

    # downloading from s3
    s3_handler = S3Handler(bucket_name="deep-bucket-s3")
    s3_handler.download_folder(
        "checkpoints_test",
        "checkpoints",
    )
    checkpoint_path = "./checkpoints/best_model.ckpt"
    try:
        # Perform testing
        test_accuracy = test_model(checkpoint_path)
        logger.info(f"Test completed successfully with accuracy: {test_accuracy:.2%}")

        # Example inference
        logger.info("Running inference on a single test image...")
        dummy_image = torch.randn(1, 28, 28)  # Replace with actual test image
        predicted_class = infer(checkpoint_path, dummy_image)
        logger.info(f"Inference result: Predicted class {predicted_class}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
