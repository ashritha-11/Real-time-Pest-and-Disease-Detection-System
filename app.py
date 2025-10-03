import streamlit as st
import os
import hashlib
from datetime import datetime
from supabase import create_client, Client
from PIL import Image
import numpy as np
import tensorflow as tf

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
        # Test connection
        _ = supabase.table("farmers").select("*").limit(1).execute()
        connection_status = "✅ Connected to Supabase"
except Exception as e:
    supabase = None
    connection_status = f"❌ Supabase connection failed: {e}"

# --------------------------
# Hashing
# --------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --------------------------
# Auth Functions
# --------------------------
def login_user(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            resp = supabase.table(table).select("*").eq("username", username).execute()
            if resp.data:
                user = resp.data[0]
                if user.get("password") == hash_password(password):
                    return True, user
        except Exception as e:
            st.error(f"Login error: {e}")
    return False, None

def register_user(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            supabase.table(table).insert({
                "username": username,
                "password": hash_password(password),
                "role": role
            }).execute()
            st.success("✅ Registered successfully!")
        except Exception as e:
            st.error(f"Registration error: {e}")
    else:
        st.warning("⚠ Supabase not available")

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
# Model (Dummy example, replace with your CNN)
# --------------------------
MODEL_PATH = "models/cnn_model.h5"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except:
        model = None

def predict_image(uploaded_file):
    # Replace with real model prediction
    if model:
        img = Image.open(uploaded_file).resize((224,224)).convert("RGB")
        arr = np.array(img)/255.0
        arr = arr.reshape((1,)+arr.shape)
        probs = model.predict(arr)[0]
        idx = probs.argmax()
        label = "Healthy" if idx==0 else ("Pest-Affected" if idx==1 else "Disease-Affected")
        return label, float(probs[idx])
    return "Pest Detected", 0.92

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌱 Real-time Pest & Disease Detection System")
st.info(f"Supabase Status: {connection_status}")

menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("Menu", menu)
role = st.sidebar.radio("Role", ["Farmer", "Admin"])

# --------------------------
# Session State Initialization
# --------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None

# --------------------------
# Login
# --------------------------
if choice == "Login":
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, user = login_user(username, password, role)
        if success:
            st.session_state["user"] = username
            st.session_state["role"] = role
            # Store user_id safely
            if role.lower() == "farmer":
                st.session_state["user_id"] = user.get("farmer_id")
            else:
                st.session_state["user_id"] = user.get("admin_id")
            st.success(f"Welcome, {username}! You are logged in as {role}.")
        else:
            st.error("❌ Invalid login credentials")

# --------------------------
# Register
# --------------------------
elif choice == "Register":
    st.subheader("📝 Register")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Register"):
        register_user(username, password, role)

# --------------------------
# Upload & Detect
# --------------------------
elif choice == "Upload & Detect":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📤 Upload Crop Image for Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("Run Detection"):
                pred, conf = predict_image(uploaded_file)
                st.success(f"Prediction: {pred} (Confidence: {conf:.2f})")
                save_detection(st.session_state["user_id"], pred, conf, uploaded_file.name)

# --------------------------
# History
# --------------------------
elif choice == "History":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📜 Detection History")
        if supabase and st.session_state["role"].lower()=="farmer":
            try:
                resp = supabase.table("detection_records").select("*").eq("farmer_id", st.session_state["user_id"]).order("timestamp", desc=True).execute()
                if resp.data:
                    for rec in resp.data:
                        st.write(f"🗓 {rec['timestamp']} → {rec['prediction']} ({rec['confidence']})")
                else:
                    st.info("No history found.")
            except Exception as e:
                st.error(f"History error: {e}")
        elif st.session_state["role"].lower()=="admin" and supabase:
            try:
                resp = supabase.table("detection_records").select("*").order("timestamp", desc=True).execute()
                if resp.data:
                    for rec in resp.data:
                        st.write(f"👨‍🌾 Farmer ID {rec['farmer_id']} → {rec['prediction']} ({rec['confidence']})")
                else:
                    st.info("No history found.")
            except Exception as e:
                st.error(f"History error: {e}")

# --------------------------
# Logout
# --------------------------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None
    st.experimental_rerun()
