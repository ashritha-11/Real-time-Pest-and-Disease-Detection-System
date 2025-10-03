# predict_test.py
import json, numpy as np, tensorflow as tf, cv2
MODEL_PATH = "models/pest_disease_model.h5"
LABELS_PATH = "models/class_indices.json"
IMG_PATH = "some_test_image.jpg"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_label = {v: k for k, v in class_indices.items()}

# preprocess same way as training
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224,224))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, 0)

preds = model.predict(img)
i = int(np.argmax(preds[0]))
confidence = float(np.max(preds[0]))
print("Predicted:", idx_to_label[i], "confidence:", confidence)
