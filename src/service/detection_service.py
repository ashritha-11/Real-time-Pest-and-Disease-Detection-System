# service/detection_service.py
import json
import numpy as np
import tensorflow as tf
from utils.image_utils import preprocess_image   # see below
from dao.detection_dao import add_detection      # your Supabase DAO or local DAO

MODEL_PATH = "models/pest_disease_model.h5"
LABELS_PATH = "models/class_indices.json"

model = None
labels = {}

if tf.io.gfile.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH} - run training first.")

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}  # invert: idx -> name

def predict_and_store(farmer_id, device_id, image_path):
    img = preprocess_image(image_path)   # returns shape (1,224,224,3)
    preds = model.predict(img)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    prediction = labels.get(class_idx, str(class_idx))

    # Save to DB/storage
    add_detection(farmer_id, class_idx, device_id, image_path, confidence)
    return prediction, confidence
