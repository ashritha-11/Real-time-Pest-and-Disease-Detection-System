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
# Streamlit UI Setup
# --------------------------
st.set_page_config(page_title="🌿 Pest & Disease Detection System", layout="wide")

# --------------------------
# Dark Mode Colors
# --------------------------
bg_color = "#000000"
header_bg = "#1a1a1a"
header_text = "#00ff99"
input_bg = "#222222"
input_text = "#ffffff"
btn_bg = "#00cc66"
btn_hover = "#00994d"
info_bg = "#111111"
info_text = "#00ff99"

# Apply CSS for dark mode
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}
h1, h2, .stMarkdown h1, .stMarkdown h2 {{
    background-color: {header_bg};
    color: {header_text};
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}}
.stTextInput input, .stPasswordInput input {{
    background-color: {input_bg};
    color: {input_text};
    border-radius: 10px;
    padding: 10px;
}}
.stButton button {{
    background-color: {btn_bg};
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
}}
.stButton button:hover {{
    background-color: {btn_hover};
}}
.stInfo, .stWarning, .stError {{
    background-color: {info_bg} !important;
    color: {info_text} !important;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Header
# --------------------------
st.markdown(f"## 🌿 Real-time Pest & Disease Detection System")
st.info(f"Supabase Status: {connection_status}")

# --------------------------
# Session state
# --------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None

# --------------------------
# Sidebar Menu
# --------------------------
menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("📋 Menu", menu)
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
            st.success(f"✅ Welcome, {username}! Logged in as {role}.")
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
                prediction, confidence = predict_image(save_path)

                # Prediction colors for dark mode
                colors = {"Healthy": "#00ff99", "Not Healthy": "#ffcc00",
                          "Pest_Affected": "#ff3300", "Disease_Affected": "#ff6600"}
                st.markdown(f"<p style='color:{colors.get(prediction,'white')};"
                            f"font-weight:bold'>Prediction: {prediction} ({confidence*100:.1f}%)</p>",
                            unsafe_allow_html=True)

                save_detection(st.session_state["user_id"], prediction, confidence, save_path)

# ---------- History ----------
elif choice == "History":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📜 Detection History")
        if supabase:
            try:
                if st.session_state["role"].lower() == "admin":
                    farmers_resp = supabase.table("farmers").select("*").execute()
                    farmers_list = [f["username"] for f in farmers_resp.data] if farmers_resp.data else []
                    selected_farmer = st.selectbox("Filter by Farmer", ["All"] + farmers_list)
                    query = supabase.table("detection_records").select(
                        "id, farmer_id, prediction, confidence, image_url, timestamp, farmers(username)"
                    ).order("timestamp", desc=True)
                    if selected_farmer != "All":
                        farmer_id = next((f["farmer_id"] for f in farmers_resp.data if f["username"]==selected_farmer), None)
                        query = query.eq("farmer_id", farmer_id)
                    resp = query.execute()
                    records = resp.data if resp.data else []
                    for rec in records:
                        farmer_name = rec.get("farmers", {}).get("username", "Unknown")
                        st.markdown(f"**Farmer:** {farmer_name}  \n**Prediction:** {rec['prediction']}  \n**Confidence:** {rec['confidence']*100:.1f}%  \n**Timestamp:** {rec['timestamp']}")
                        if rec.get("image_url"):
                            st.image(rec["image_url"], width=200)
                        st.markdown("---")
                    if records:
                        df = pd.DataFrame([{"Farmer": rec.get("farmers", {}).get("username", ""),
                                            "Prediction": rec["prediction"],
                                            "Confidence": rec["confidence"],
                                            "Image URL": rec["image_url"],
                                            "Timestamp": rec["timestamp"]} for rec in records])
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV Report", csv, file_name="detection_report.csv")
                else:
                    resp = supabase.table("detection_records").select("*").eq("farmer_id", st.session_state["user_id"]).order("timestamp", desc=True).execute()
                    if resp.data:
                        for rec in resp.data:
                            st.markdown(f"**Prediction:** {rec['prediction']}  \n**Confidence:** {rec['confidence']*100:.1f}%  \n**Timestamp:** {rec['timestamp']}")
                            if rec.get("image_url"):
                                st.image(rec["image_url"], width=200)
                            st.markdown("---")
                    else:
                        st.info("No records found.")
            except Exception as e:
                st.error(f"History error: {e}")
        else:
            st.warning("⚠ Supabase not connected")

# ---------- Logout ----------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()
