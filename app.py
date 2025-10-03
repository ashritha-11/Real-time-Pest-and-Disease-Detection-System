import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import json
from supabase import create_client, Client
from datetime import datetime
import hashlib

# ---------------- Streamlit Secrets / Supabase ----------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Paths ----------------
MODEL_PATH = "models/pest_disease_model.h5"  # Keras model
CLASS_INDICES_PATH = "models/class_indices.json"

# ---------------- Load Model ----------------
model = load_model(MODEL_PATH)

# ---------------- Load Class Indices ----------------
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# ---------------- Transform / Preprocess ----------------
IMG_SIZE = (224, 224)

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# ---------------- Helper Functions ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        supabase.table("users").insert({
            "username": username,
            "password": hash_password(password),
            "created_at": datetime.now().isoformat()
        }).execute()
        st.success("✅ Registration successful! Please login.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

def login_user(username, password):
    try:
        resp = supabase.table("users").select("*").eq("username", username).execute()
        if resp.data:
            stored_pw = resp.data[0]["password"]
            if stored_pw == hash_password(password):
                return True
        return False
    except Exception as e:
        st.error(f"❌ Supabase Error: {e}")
        return False

def save_detection(user, image_file, label):
    try:
        supabase.table("detections").insert({
            "username": user,
            "image_name": image_file.name,
            "prediction": label,
            "timestamp": datetime.now().isoformat()
        }).execute()
        st.info("✅ Detection saved!")
    except Exception as e:
        st.error(f"❌ Supabase Error: {e}")

def show_history(username):
    st.subheader("📊 Your Detection History")
    try:
        history = supabase.table("detections").select("*").eq("username", username).order("timestamp", desc=True).execute()
        if history.data:
            for item in history.data:
                st.write(f"Image: {item['image_name']}, Prediction: {item['prediction']}, Time: {item['timestamp']}")
        else:
            st.info("No detections yet.")
    except Exception as e:
        st.error(f"❌ Supabase Error: {e}")

# ---------------- Streamlit UI ----------------
st.title("🌿 Pest & Disease Detection System")

# ---------------- Session State ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- Sidebar Menu ----------------
if not st.session_state.logged_in:
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        st.subheader("Create a New Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            if username and password:
                register_user(username, password)
            else:
                st.warning("Please fill all fields")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome {username}!")
            else:
                st.error("❌ Invalid username or password")

else:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

    # ---------------- Upload Image & Predict ----------------
    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(img)
        pred_probs = model.predict(img_array)
        pred_index = np.argmax(pred_probs)
        label = class_indices[str(pred_index)]

        st.success(f"Detected: **{label}**")

        if st.button("Save Detection"):
            save_detection(st.session_state.username, uploaded_file, label)

    # ---------------- Show Detection History ----------------
    show_history(st.session_state.username)
