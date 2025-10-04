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
# Advanced Pest/Disease Highlighting (Tiny Spots + Heatmap)
# --------------------------
def segment_and_highlight_advanced(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Color-based masks
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([180, 255, 90])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    lower_lab = np.array([0, 120, 0])
    upper_lab = np.array([255, 145, 255])
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

    # Texture-based mask
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    _, mask_lap = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
    combined_mask = cv2.bitwise_or(combined_mask, mask_lap)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Contours and bounding boxes
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pest_count = 0
    heatmap = np.zeros_like(gray, dtype=np.float32)

    for c in contours:
        if cv2.contourArea(c) > 5:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(img_rgb, "Pest/Disease", (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            pest_count += 1

            # Add heat intensity
            heatmap[y:y+h, x:x+w] += 1

    # Normalize heatmap and overlay
    if np.max(heatmap) > 0:
        heatmap_norm = np.uint8(255 * heatmap / np.max(heatmap))
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(img_rgb, 0.7, heatmap_color, 0.3, 0)
    else:
        overlayed = img_rgb

    return Image.fromarray(overlayed), pest_count

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
        top_indices = probs.argsort()[-2:][::-1]
        top_conf = [probs[i] for i in top_indices]
        top_labels = [idx_to_label.get(i, "Unknown") for i in top_indices]

        if top_conf[0] < threshold:
            label = "Not Healthy"
            confidence = top_conf[0]
            st.warning(f"⚠ Low confidence prediction. Possible issues: {', '.join(top_labels)}")
        else:
            label = top_labels[0]
            confidence = top_conf[0]

        return label, confidence
    return "Unknown", 0.0

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🌱 Pest & Disease Detection System", layout="wide")
st.title("🌱 Real-time Pest & Disease Detection System")
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
                prediction, confidence = predict_image(save_path)

                if prediction == "Healthy":
                    st.success(f"✅ Prediction: {prediction} (Confidence: {confidence*100:.1f}%)")
                elif prediction == "Not Healthy":
                    st.warning(f"⚠ Prediction: {prediction} (Confidence: {confidence*100:.1f}%)")
                elif prediction == "Pest_Affected":
                    st.error(f"🐛 Prediction: {prediction} (Confidence: {confidence*100:.1f}%)")
                elif prediction == "Disease_Affected":
                    st.error(f"🍂 Prediction: {prediction} (Confidence: {confidence*100:.1f}%)")
                else:
                    st.info(f"❔ Prediction: {prediction} (Confidence: {confidence*100:.1f}%)")

                # Advanced segmentation + heatmap
                highlighted_img, pest_count = segment_and_highlight_advanced(save_path)
                st.subheader(f"🔹 Pest / Disease Highlights (Count: {pest_count})")
                st.image(highlighted_img, use_container_width=True)

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
                        st.markdown(f"""
                            **Farmer:** {farmer_name}  
                            **Prediction:** {rec['prediction']}  
                            **Confidence:** {rec['confidence']*100:.1f}%  
                            **Timestamp:** {rec['timestamp']}  
                        """)
                        if rec.get("image_url"):
                            st.image(rec["image_url"], width=200)
                        st.markdown("---")

                    if records:
                        df = pd.DataFrame([
                            {
                                "Farmer": rec.get("farmers", {}).get("username", ""),
                                "Prediction": rec["prediction"],
                                "Confidence": rec["confidence"],
                                "Image URL": rec["image_url"],
                                "Timestamp": rec["timestamp"]
                            }
                            for rec in records
                        ])
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
