import os
import io
import json
import numpy as np

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

model = None
class_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    from tf_keras.models import load_model
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)
    print(f"✅ Number of classes: {len(class_names)}")
    yield

app = FastAPI(title="Plant Disease Prediction API", lifespan=lifespan)

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)

        preds = model.predict(img_array, verbose=0)
        class_index = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_index])

        if isinstance(class_names, list):
            predicted_class = class_names[class_index]
        else:
            predicted_class = class_names.get(str(class_index), str(class_index))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
