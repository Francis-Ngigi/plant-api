import os
import io
import json
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tf_keras.models import load_model
from PIL import Image

# -----------------------
# FastAPI app setup
# -----------------------
app = FastAPI(title="Plant Disease Prediction API")

# -----------------------
# Paths — files sit next to main.py in the project root
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

# -----------------------
# Load model and classes at startup
# -----------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_PATH}")

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)
print(f"✅ Number of classes: {len(class_names)}")

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