import os
import io
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_v2.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

app = FastAPI(title="Plant Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = None
load_error = None

def load_resources():
    global model, class_names, load_error
    if load_error:
        raise Exception(f"Model failed to load: {load_error}")
    if model is None:
        try:
            print(f"TensorFlow version: {tf.__version__}")
            print("Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded.")
        except Exception as e:
            load_error = str(e)
            print(f"❌ Model load error: {e}")
            raise
    if class_names is None:
        with open(CLASS_PATH, "r") as f:
            class_names = json.load(f)
        print(f"✅ Classes loaded: {len(class_names)}")

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "load_error": load_error
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_resources()
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
        print(f"❌ Predict error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)
