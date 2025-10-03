import tensorflow as tf
from PIL import Image
import numpy as np
from dao.detection_dao import add_detection

MODEL_PATH = "models/cnn_model.h5"
model = None

if tf.io.gfile.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

def predict_image(path: str):
    img = Image.open(path).resize((224,224)).convert("RGB")
    arr = np.array(img)/255.0
    arr = arr.reshape((1,)+arr.shape)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    label = "Healthy" if idx==0 else ("Pest-Affected" if idx==1 else "Disease-Affected")
    confidence = float(preds[idx])
    return label, confidence

def predict_and_store(farmer_id, image_path):
    label, confidence = predict_image(image_path)
    add_detection(farmer_id, label, confidence, image_path)
    return label, confidence
