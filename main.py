import io
import base64
from pathlib import Path
from typing import Annotated
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fasthtml.common import (
    Html,
    Head,
    Title,
    Script,
    Body,
    Div,
    Form,
    Input,
    Img,
    P,
    to_xml,
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    Button,
    Badge,
    Progress,
    Separator,
    Alert,
    AlertTitle,
    AlertDescription,
)
from PIL import Image
from loguru import logger
from src.utils.aws_s3_services import S3Handler
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Classifier with FastHTML",
    description="Classify handwritten digits using a trained model with a FastHTML UI",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
log_dir = "/tmp/logs"
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger.add(f"{log_dir}/inference.log", rotation="1 MB", level="INFO", enqueue=False)

# Model setup
MODEL_PATH = "./checkpoints/traced_model.pt"
LABELS = [str(i) for i in range(10)]


class MNISTClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference will run on device: {self.device}")

        # Load model
        self.model = self.load_model()
        self.model.eval()

        # Define preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def load_model(self):
        if not Path(self.model_path).exists():
            logger.error(f"Traced model not found: {self.model_path}")
            raise FileNotFoundError(f"Traced model not found: {self.model_path}")

        logger.info(f"Loading TorchScript model from: {self.model_path}")
        return torch.jit.load(self.model_path).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return {LABELS[idx]: float(prob) for idx, prob in enumerate(probabilities)}


classifier = None


@app.on_event("startup")
async def startup_event():
    global classifier

    if not Path(MODEL_PATH).exists():
        logger.info("Downloading model from S3...")
        s3_handler = S3Handler(bucket_name="deep-bucket-s3")
        s3_handler.download_folder("checkpoints_test", "checkpoints")

    logger.info("Initializing the MNIST Classifier...")
    classifier = MNISTClassifier(model_path=MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    """
    Serve the FastHTML-based UI for image upload and prediction.
    """
    try:
        content = Html(
            Head(
                Title("MNIST Classifier"),
                ShadHead(tw_cdn=True, theme_handle=True),
                Script(
                    src="https://unpkg.com/htmx.org@2.0.3",
                    integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                    crossorigin="anonymous",
                ),
            ),
            Body(
                Div(
                    Div(
                        Div(
                            CardHeader(
                                CardTitle(
                                    "MNIST Classifier 🖌️",
                                    cls="text-white text-2xl font-bold",
                                ),
                                Badge(
                                    "AI Powered",
                                    cls="bg-yellow-400 text-black px-3 py-1 rounded-full",
                                ),
                            ),
                            cls="flex items-center justify-between mb-4",
                        ),
                        CardContent(
                            Form(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        cls="block w-full text-sm text-gray-200 border border-gray-500 rounded-lg bg-gray-800 focus:ring-yellow-500 focus:border-yellow-500",
                                    ),
                                    Button(
                                        "Classify Image",
                                        type="submit",
                                        cls="mt-4 bg-yellow-500 text-black font-bold px-4 py-2 rounded-lg hover:bg-yellow-400 transition",
                                    ),
                                    cls="space-y-4",
                                ),
                                enctype="multipart/form-data",
                                hx_post="/classify",
                                hx_target="#result",
                            ),
                        ),
                        Div(id="result", cls="mt-6"),
                        cls="p-6 bg-gray-900 shadow-lg rounded-lg",
                    ),
                    cls="container mx-auto mt-20 max-w-lg",
                ),
                cls="bg-black text-white min-h-screen flex items-center justify-center",
            ),
        )
        return to_xml(content)
    except Exception as e:
        # Log detailed error and return an Alert on failure
        print(f"Error rendering UI: {e}")
        error_alert = Html(
            Body(
                Alert(
                    AlertTitle("Rendering Error"),
                    AlertDescription(str(e)),
                    variant="destructive",
                    cls="mt-4",
                )
            )
        )
        return to_xml(error_alert)


@app.post("/classify", response_class=HTMLResponse)
async def ui_handle_classify(file: Annotated[bytes, File()]):
    try:
        image = Image.open(io.BytesIO(file)).convert("L")
        predictions = classifier.predict(image)
        top_prediction = max(predictions, key=predictions.get)
        confidence = predictions[top_prediction]

        # Create a brighter UI result
        result = Div(
            Div(
                Badge(
                    f"Prediction: {top_prediction} ({confidence:.1%})",
                    cls="w-full text-lg bg-green-500 text-black font-bold px-4 py-2 rounded-lg",
                ),
                Progress(
                    value=int(confidence * 100),
                    cls="mt-4 h-4 rounded-lg bg-gray-700",
                ),
                cls="space-y-4",
            ),
            cls="mt-6",
        )
        return to_xml(result)
    except Exception as e:
        error_alert = Alert(
            AlertTitle("Classification Error"),
            AlertDescription(str(e)),
            variant="destructive",
            cls="mt-4",
        )
        return to_xml(error_alert)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": classifier is not None}
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server with FastHTML...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
        workers=1,
        debug=True,
        use_colors=True,
    )
