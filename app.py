import io
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from src.utils.aws_s3_services import S3Handler

app = FastAPI()

# Add CORS Middleware
origins = ["http://localhost", "http://localhost:8080"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
log_dir = "/tmp/logs"
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger.add(f"{log_dir}/inference.log", rotation="1 MB", level="INFO", enqueue=False)

# Model Path
MODEL_PATH = "./checkpoints/traced_model.pt"


class MNISTClassifier:
    def __init__(self, model_path: str):
        """
        Initializes the MNIST classifier for inference.
        """

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference will run on device: {self.device}")

        # Load the TorchScript model
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
        Loads the TorchScript model for inference.
        """
        if not Path(self.model_path).exists():
            logger.error(f"Traced model not found: {self.model_path}")
            raise FileNotFoundError(f"Traced model not found: {self.model_path}")

        logger.info(f"Loading TorchScript model from: {self.model_path}")
        return torch.jit.load(self.model_path).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image):
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


# Initialize the classifier
classifier = None


@app.on_event("startup")
async def startup_event():
    """
    Downloads the model from S3 and initializes the classifier during app startup.
    """
    global classifier

    # Check if the model exists locally; if not, download it from S3
    if not Path(MODEL_PATH).exists():
        logger.info("Downloading model from S3...")
        s3_handler = S3Handler(bucket_name="deep-bucket-s3")
        s3_handler.download_folder("checkpoints_test", "checkpoints")

    # Initialize the classifier
    logger.info("Initializing the MNIST Classifier...")
    classifier = MNISTClassifier(model_path=MODEL_PATH)


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]):
    """
    API endpoint to perform prediction on uploaded image.

    Args:
        file: The uploaded image file.

    Returns:
        JSON response containing the predicted class probabilities.
    """
    try:
        # Read the uploaded image
        file_b = await file.read()
        img = Image.open(io.BytesIO(file_b))
        img = img.convert("L")  # MNIST expects grayscale images

        # Perform prediction
        prediction = classifier.predict(img)

        return JSONResponse(content={"predictions": prediction})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to process the image"}, status_code=500
        )


@app.get("/health")
async def health_check():
    """
    API endpoint to check the health of the service.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting the FastAPI server...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Exposes the application to external traffic
        port=8000,  # Sets the server port
        reload=False,  # Auto-reloads the server during development
        workers=2,  # Number of worker processes for handling requests
        timeout_keep_alive=150,  # Time (in seconds) to keep the connection alive
        access_log=True,  # Enables access logs
        log_level="info",  # Logging level: debug, info, warning, error, critical
        lifespan="on",  # Ensures startup and shutdown events are properly triggered
    )
