import streamlit as st
import hashlib
from datetime import datetime
from supabase import create_client, Client
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import json
import cv2

# --------------------------
# Supabase Setup
# --------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

supabase: Client | None = None
connection_status = "❌ Not Connected"

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        _ = supabase.table("farmers").select("*").limit(1).execute()
        connection_status = "✅ Connected to Supabase"
    else:
        connection_status = "❌ Secrets missing"
except Exception as e:
    connection_status = f"❌ Supabase connection failed: {e}"
    supabase = None

# --------------------------
# Hashing
# --------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --------------------------
# Auth Functions
# --------------------------
def register_user(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            supabase.table(table).insert({
                "username": username,
                "password": hash_password(password),
                "role": role
            }).execute()
            st.success(f"✅ {role} registered successfully!")
        except Exception as e:
            st.error(f"Registration error: {e}")
    else:
        st.warning("⚠ Supabase not available")

def login_user(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            resp = supabase.table(table).select("*").eq("username", username).execute()
            if resp.data:
                user = resp.data[0]
                if user["password"] == hash_password(password):
                    return user
        except Exception as e:
            st.error(f"Login error: {e}")
    return None

# --------------------------
# Detection Save
# --------------------------
def save_detection(farmer_id, prediction, confidence, image_url):
    if supabase:
        try:
            supabase.table("detection_records").insert({
                "farmer_id": farmer_id,
                "prediction": prediction,
                "confidence": confidence,
                "image_url": image_url,
                "timestamp": datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            st.error(f"Insert error: {e}")

# --------------------------
# ML Model Setup
# --------------------------
MODEL_PATH = "models/cnn_model.h5"
LABELS_PATH = "models/class_indices.json"
model = None
idx_to_label = {0: "Healthy", 1: "Pest_Affected", 2: "Disease_Affected"}

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        model = None

if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r") as f:
            class_indices = json.load(f)
        idx_to_label = {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.warning(f"⚠ Could not load class indices: {e}")

# --------------------------
# Prediction Function
# --------------------------
def predict_image(file_path):
    """Predicts class using CNN."""
    if model:
        img = Image.open(file_path).convert("RGB")
        arr = np.array(img)
        arr = tf.image.resize(arr, (224, 224))
        arr = np.expand_dims(arr, axis=0)
        arr = arr / 255.0

        probs = model.predict(arr, verbose=0)[0]
        top_index = np.argmax(probs)
        label = idx_to_label.get(top_index, "Healthy")
        confidence = probs[top_index]

        return label, confidence
    return "Healthy", 0.0

# --------------------------
# Improved OpenCV Pest Highlight & Count
# --------------------------
def highlight_pests_cv2(image_path):
    """Returns image with pests highlighted and pest count."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV to detect dark/brown spots
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold for dark/brown spots (typical pests/disease)
    lower = np.array([0, 30, 0])
    upper = np.array([50, 255, 120])
    mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pest_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 2000:  # ignore tiny spots and huge blobs
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(img_rgb, "Pest/Disease", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            pest_count += 1

    return Image.fromarray(img_rgb), pest_count

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🌱 Pest & Disease Detection", layout="wide")
st.title("🌱 Pest & Disease Detection System")
st.info(f"Supabase Status: {connection_status}")

if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None

menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("Menu", menu)
role = st.sidebar.radio("Role", ["Farmer", "Admin"])

# ---------- Login ----------
if choice == "Login":
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password, role)
        if user:
            st.session_state["user"] = username
            st.session_state["role"] = role
            st.session_state["user_id"] = user[f"{role.lower()}_id"]
            st.success(f"Welcome, {username}! Logged in as {role}.")
        else:
            st.error("❌ Invalid credentials or user not found.")

# ---------- Register ----------
elif choice == "Register":
    st.subheader("📝 Register")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Register"):
        register_user(username, password, role)

# ---------- Upload & Detect ----------
elif choice == "Upload & Detect":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    elif st.session_state["role"].lower() == "farmer":
        st.subheader("📤 Upload Crop Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        if uploaded_file:
            save_path = f"{st.session_state['user']}_{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(save_path, use_container_width=True)

            if st.button("Run Detection"):
                # Step 1: CNN prediction
                prediction, confidence = predict_image(save_path)

                # Step 2: OpenCV pest highlight & count
                highlighted_img, pest_count = highlight_pests_cv2(save_path)

                # Step 3: Override if CNN predicts Healthy but pests exist
                if prediction == "Healthy" and pest_count > 0:
                    prediction = "Pest_Affected"
                    confidence = 0.6  # moderate confidence

                # Step 4: Display prediction
                display_map = {
                    "Healthy": ("✅ Healthy", "success"),
                    "Pest_Affected": ("🐛 Pest Affected", "error"),
                    "Disease_Affected": ("🍂 Disease Affected", "error")
                }

                text, style = display_map[prediction]
                message = f"{text} (Confidence: {confidence*100:.1f}%)"

                if style == "success":
                    st.success(message)
                else:
                    st.error(message)

                # Step 5: Show pest highlights if any
                if pest_count > 0:
                    st.subheader(f"🔹 Pest / Disease Highlights (Count: {pest_count})")
                    st.image(highlighted_img, use_container_width=True)

                save_detection(st.session_state["user_id"], prediction, confidence, save_path)

# ---------- History ----------
elif choice == "History":
    st.subheader("📜 Detection History")
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    elif supabase:
        try:
            records_resp = supabase.table("detection_records").select("*").eq("farmer_id", st.session_state["user_id"]).order("timestamp", desc=True).execute()
            records = records_resp.data if records_resp.data else []

            for rec in records:
                st.markdown(f"""
                    **Prediction:** {rec['prediction']}  
                    **Confidence:** {rec['confidence']*100:.1f}%  
                    **Timestamp:** {rec['timestamp']}  
                """)
                if rec.get("image_url"):
                    st.image(rec["image_url"], width=200)
                st.markdown("---")
        except Exception as e:
            st.error(f"History error: {e}")
    else:
        st.warning("⚠ Supabase not connected")

# ---------- Logout ----------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None
    st.rerun()
