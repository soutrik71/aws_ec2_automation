import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from loguru import logger
from src.model import LitEfficientNet
from src.utils.aws_s3_services import S3Handler

# Configure Loguru for logging
logger.add("logs/inference.log", rotation="1 MB", level="INFO")


class MNISTClassifier:
    def __init__(self, checkpoint_path="./checkpoints/best_model.ckpt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference will run on device: {self.device}")

        # Load the model
        self.model = self.load_model()
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.labels = [str(i) for i in range(10)]  # MNIST labels are 0-9

    def load_model(self):
        """
        Loads the model checkpoint for inference.
        """
        if not Path(self.checkpoint_path).exists():
            logger.error(f"Checkpoint not found: {self.checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        return LitEfficientNet.load_from_checkpoint(self.checkpoint_path).to(
            self.device
        )

    @torch.no_grad()
    def predict(self, image):
        """
        Perform inference on a single image.

        Args:
            image: Input image in PIL format.

        Returns:
            dict: Predicted class probabilities.
        """
        if image is None:
            logger.error("No image provided for prediction.")
            return None

        # Convert to tensor and preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Perform inference
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Map probabilities to labels
        return {self.labels[idx]: float(prob) for idx, prob in enumerate(probabilities)}


# Instantiate the classifier
checkpoint_path = "./checkpoints/best_model.ckpt"

# Download checkpoint from S3 (if needed)
s3_handler = S3Handler(bucket_name="deep-bucket-s3")
s3_handler.download_folder(
    "checkpoints_test",
    "checkpoints",
)

classifier = MNISTClassifier(checkpoint_path=checkpoint_path)

# Define Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(height=160, width=160, image_mode="L", type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="MNIST Classifier",
    description="Upload a handwritten digit image to classify it (0-9).",
)

if __name__ == "__main__":
    demo.launch(share=True)
