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
# Streamlit Page Config & Styling
# --------------------------
st.set_page_config(page_title="🌱 Pest & Disease Detection", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f5fff5;
        }
        .main {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0px 4px 10px rgba(0, 128, 0, 0.2);
        }
        .stButton>button {
            background-color: #2e8b57;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #228b22;
        }
        .prediction-box {
            background-color: #f0fff0;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 3px 8px rgba(0, 100, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

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
# Utility Functions
# --------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
# Model Setup
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
# Prediction Function (Fixed & Enhanced)
# --------------------------
def predict_image(file_path, threshold_healthy=0.7):
    if model is None:
        st.error("❌ Model not loaded")
        return "Unknown", 0.0

    try:
        img = Image.open(file_path).convert("RGB")
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = idx_to_label.get(idx, "Unknown")

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mean_color = cv2.mean(cv_img)[:3]
        red, green, blue = mean_color
        is_brownish = (red > 90 and red < 180) and (green > 60 and green < 160) and (blue < 130)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray < 60) / gray.size

        if label == "Healthy":
            if confidence < threshold_healthy:
                label = "Disease_Affected"
            elif is_brownish:
                label = "Disease_Affected"
            elif dark_ratio > 0.10:
                label = "Pest_Affected"

        return label, confidence

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Unknown", 0.0

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌿 Real-time Pest & Disease Detection System")
st.caption("Powered by Deep Learning & Supabase")
st.info(f"Supabase Connection: {connection_status}")

if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None

menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("📋 Menu", menu)
role = st.sidebar.radio("👤 Role", ["Farmer", "Admin"])

# ---------- Login ----------
if choice == "Login":
    with st.container():
        st.subheader("🔐 Login to Your Account")
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
    with st.container():
        st.subheader("📝 Create a New Account")
        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        if st.button("Register"):
            register_user(username, password, role)

# ---------- Upload & Detect ----------
elif choice == "Upload & Detect":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    elif st.session_state["role"].lower() == "farmer":
        st.subheader("📤 Upload Your Crop Leaf Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        if uploaded_file:
            save_path = f"{st.session_state['user']}_{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.image(save_path, caption="Uploaded Leaf", use_container_width=True)

            with col2:
                if st.button("Run Detection", use_container_width=True):
                    prediction, confidence = predict_image(save_path)
                    with st.container():
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        if prediction == "Healthy":
                            st.success(f"✅ **Prediction:** {prediction}\n\n**Confidence:** {confidence*100:.1f}%")
                        elif prediction == "Pest_Affected":
                            st.error(f"🐛 **Prediction:** {prediction}\n\n**Confidence:** {confidence*100:.1f}%")
                        elif prediction == "Disease_Affected":
                            st.warning(f"🍂 **Prediction:** {prediction}\n\n**Confidence:** {confidence*100:.1f}%")
                        else:
                            st.info(f"❔ Prediction: {prediction}")
                        st.markdown('</div>', unsafe_allow_html=True)
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
                        st.markdown(f"""
                            **Farmer:** {rec.get("farmers", {}).get("username", "Unknown")}  
                            **Prediction:** {rec['prediction']}  
                            **Confidence:** {rec['confidence']*100:.1f}%  
                            **Timestamp:** {rec['timestamp']}  
                        """)
                        if rec.get("image_url"):
                            st.image(rec["image_url"], width=200)
                        st.markdown("---")

                    if records:
                        df = pd.DataFrame([{
                            "Farmer": rec.get("farmers", {}).get("username", ""),
                            "Prediction": rec["prediction"],
                            "Confidence": rec["confidence"],
                            "Image URL": rec["image_url"],
                            "Timestamp": rec["timestamp"]
                        } for rec in records])
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV Report", csv, file_name="detection_report.csv")
                else:
                    resp = supabase.table("detection_records").select("*").eq("farmer_id", st.session_state["user_id"]).order("timestamp", desc=True).execute()
                    if resp.data:
                        for rec in resp.data:
                            st.markdown(f"""
                                **Prediction:** {rec['prediction']}  
                                **Confidence:** {rec['confidence']*100:.1f}%  
                                **Timestamp:** {rec['timestamp']}  
                            """)
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
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None
    st.rerun()
