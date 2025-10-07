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
            st.success("✅ Detection saved to Supabase!")
        except Exception as e:
            st.error(f"Insert error: {e}")
    else:
        st.warning("⚠ Supabase not available")

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
def predict_image(file_path, threshold=0.7):
    if model:
        img = Image.open(file_path).convert("RGB")
        arr = np.array(img)
        arr = tf.image.resize(arr, (224, 224))
        arr = np.expand_dims(arr, axis=0)
        arr = arr / 255.0
        probs = model.predict(arr, verbose=0)[0]
        idx = probs.argmax()
        confidence = float(probs[idx])
        label = idx_to_label.get(idx, "Unknown")
        if label == "Healthy" and confidence < threshold:
            label = "Not Healthy"
        return label, confidence
    return "Unknown", 0.0

# --------------------------
# Streamlit UI with Auto Theme
# --------------------------
st.set_page_config(page_title="🌿 Pest & Disease Detection System", layout="wide")

# Initialize session state
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"  # default

# Sidebar for manual theme override
st.sidebar.markdown("### 🎨 Theme")
manual_theme = st.sidebar.radio("Override Theme (optional):", ["Auto", "Light", "Dark"])
st.session_state["manual_theme"] = manual_theme

# JS script to detect system dark mode
if manual_theme == "Auto":
    st.write(
        """
        <script>
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        window.parent.document.querySelector('iframe').style.backgroundColor = isDark ? '#0e1117' : '#ffffff';
        </script>
        """,
        unsafe_allow_html=True
    )
    import streamlit.components.v1 as components
    js_theme = """
        <script>
        const theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'Dark' : 'Light';
        window.parent.document.querySelector('iframe').setAttribute('data-theme', theme);
        </script>
    """
    components.html(js_theme)
    # fallback
    st.session_state["theme"] = "Dark" if st.session_state.get("manual_theme") == "Auto" else "Light"
else:
    st.session_state["theme"] = manual_theme

# Apply theme CSS
if st.session_state["theme"] == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        .stTextInput input, .stPasswordInput input { background-color: #262730; color: white; border-radius: 10px; padding: 10px; }
        .stButton button { background-color: #3b3b3b; color: white; border-radius: 10px; padding: 8px 20px; }
        .stButton button:hover { background-color: #565656; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #f8f9fa; color: black; }
        .stApp { background-color: #ffffff; }
        .stTextInput input, .stPasswordInput input { background-color: #ffffff; color: black; border-radius: 10px; padding: 10px; }
        .stButton button { background-color: #198754; color: white; border-radius: 10px; padding: 8px 20px; }
        .stButton button:hover { background-color: #157347; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------
# Header
# --------------------------
st.markdown(
    f"""
    <div style="text-align:center; padding:15px; border-radius:15px; 
    background: linear-gradient(90deg, {'#198754' if st.session_state['theme']=='Light' else '#262730'}, {'#28a745' if st.session_state['theme']=='Light' else '#3b3b3b'}); 
    color:white; font-size:26px; font-weight:bold;">
        🌿 Real-time Pest & Disease Detection System
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(f"Supabase Status: {connection_status}")

# --------------------------
# Remaining app logic: Login, Register, Upload & Detect, History, Logout
# --------------------------
# (Keep your previous logic here exactly the same)
