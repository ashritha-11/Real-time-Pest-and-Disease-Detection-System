import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Load class labels
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# Prediction function
def predict_image(image_path, model_path="models/pest_disease_model.h5"):
    # Load the trained Keras model
    model = load_model(model_path)

    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    return idx_to_class[predicted_class_idx]
