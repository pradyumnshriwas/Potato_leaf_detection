import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).parent / "models" / "potato_cnn_model.keras"
IMG_SIZE = (128, 128)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
    yield
    state.clear()


app = FastAPI(title="Potato Leaf Disease API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def preprocess(raw: bytes) -> np.ndarray:
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")
    arr = tf.image.resize(np.asarray(img, dtype=np.float32), IMG_SIZE).numpy()
    return np.expand_dims(arr, axis=0)  # no /255: the model has a Rescaling layer


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    scores = state["model"](preprocess(raw), training=False).numpy()[0]
    best = int(np.argmax(scores))
    return {
        "predicted_class": CLASS_NAMES[best],
        "confidence": float(scores[best]),
        "probabilities": {n: float(s) for n, s in zip(CLASS_NAMES, scores)},
    }