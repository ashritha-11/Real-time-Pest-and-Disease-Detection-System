import streamlit as st
import hashlib
from datetime import datetime
from supabase import create_client, Client
from PIL import Image
import numpy as np
import tensorflow as tf
import os

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
# Password Hashing
# --------------------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# --------------------------
# User Functions
# --------------------------
def register_user(username: str, password: str, role: str):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            # Check if username exists
            existing = supabase.table(table).select("*").eq("username", username).execute()
            if existing.data:
                st.warning("⚠ Username already exists")
                return False
            supabase.table(table).insert({
                "username": username,
                "password": hash_password(password),
                "role": role
            }).execute()
            st.success("✅ Registered successfully!")
            return True
        except Exception as e:
            st.error(f"Registration error: {e}")
            return False
    else:
        st.warning("⚠ Supabase not available")
        return False

def login_user(username: str, password: str, role: str):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            resp = supabase.table(table).select("*").eq("username", username).execute()
            if resp.data:
                user = resp.data[0]
                if user.get("password") == hash_password(password):
                    return user
        except Exception as e:
            st.error(f"Login error: {e}")
    return None

def get_user_id(username: str, role: str):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            resp = supabase.table(table).select("*").eq("username", username).execute()
            if resp.data:
                user = resp.data[0]
                return user[f"{role.lower()}_id"]
        except Exception as e:
            st.error(f"get_user_id error: {e}")
    return None

# --------------------------
# Detection Save
# --------------------------
def save_detection(farmer_id: int, prediction: str, confidence: float, image_url: str):
    if supabase and farmer_id:
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
        st.warning("⚠ Supabase not available or farmer not found")

# --------------------------
# Streamlit UI
# --------------------------
st.title("🌱 Real-time Pest & Disease Detection System")
st.info(f"Supabase Status: {connection_status}")

menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("Menu", menu)
role = st.sidebar.radio("Role", ["Farmer", "Admin"])

# Session variables
if "user" not in st.session_state:
    st.session_state["user"] = None
    st.session_state["role"] = None
    st.session_state["user_id"] = None

# ---------- AUTH ----------
if choice == "Register":
    st.subheader("📝 Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(username, password, role):
            st.info("Please login now.")

elif choice == "Login":
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password, role)
        if user:
            st.session_state["user"] = username
            st.session_state["role"] = role
            st.session_state["user_id"] = user[f"{role.lower()}_id"]
            st.success(f"Welcome {username}!")
        else:
            st.error("❌ Invalid credentials")

# ---------- UPLOAD & DETECT ----------
elif choice == "Upload & Detect":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📤 Upload Crop Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
            if st.button("Run Detection"):
                # Fake prediction for example (replace with ML model)
                prediction = "Pest Detected"
                confidence = 0.92
                image_url = f"https://fake-bucket/{uploaded_file.name}"
                
                st.success(f"Prediction: {prediction} ({confidence*100:.1f}%)")
                save_detection(st.session_state["user_id"], prediction, confidence, image_url)

# ---------- HISTORY ----------
elif choice == "History":
    if not st.session_state["user"]:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📜 Detection History")
        if supabase:
            try:
                records = supabase.table("detection_records").select("*").eq("farmer_id", st.session_state["user_id"]).order("timestamp", desc=True).execute()
                if records.data:
                    for rec in records.data:
                        st.write(f"🗓 {rec['timestamp']} → {rec['prediction']} ({rec['confidence']})")
                else:
                    st.info("No records found")
            except Exception as e:
                st.error(f"History error: {e}")
        else:
            st.warning("⚠ Supabase not available")
