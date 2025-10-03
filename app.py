import streamlit as st
import os
import hashlib
from datetime import datetime
from supabase import create_client, Client

# --------------------------
# Supabase Setup
# --------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

supabase = None
connection_status = "❌ Not Connected"

try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # test connection
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
def local_login(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            resp = supabase.table(table).select("*").eq("username", username).execute()
            if resp.data:
                user = resp.data[0]
                return user["password"] == hash_password(password)
        except Exception as e:
            st.error(f"Login error: {e}")
    return False

def register_user(username, password, role):
    table = "farmers" if role.lower() == "farmer" else "admins"
    if supabase:
        try:
            supabase.table(table).insert({
                "username": username,
                "password": hash_password(password)
            }).execute()
            st.success("✅ Registered successfully!")
        except Exception as e:
            st.error(f"Registration error: {e}")
    else:
        st.warning("⚠ Supabase not available")

# --------------------------
# Detection Save
# --------------------------
def save_detection(farmer_username, prediction, confidence, image_url):
    if supabase:
        try:
            supabase.table("detection_records").insert({
                "farmer_username": farmer_username,  # text instead of integer
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
# Streamlit UI
# --------------------------
st.title("🌱 Real-time Pest & Disease Detection System")
st.info(f"Supabase Status: {connection_status}")

menu = ["Login", "Register", "Upload & Detect", "History"]
choice = st.sidebar.selectbox("Menu", menu)

role = st.sidebar.radio("Role", ["Farmer", "Admin"])

if choice == "Login":
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if local_login(username, password, role):
            st.session_state["user"] = username
            st.session_state["role"] = role
            st.success(f"Welcome, {username}! You are logged in as {role}.")
        else:
            st.error("❌ Invalid login credentials")

elif choice == "Register":
    st.subheader("📝 Register")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Register"):
        register_user(username, password, role)

elif choice == "Upload & Detect":
    if "user" not in st.session_state:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📤 Upload Crop Image for Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("Run Detection"):
                # 🔮 Fake detection result (replace with ML model)
                prediction = "Pest Detected"
                confidence = 0.92
                image_url = f"https://fake-bucket/{uploaded_file.name}"

                st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
                save_detection(st.session_state["user"], prediction, confidence, image_url)

elif choice == "History":
    if "user" not in st.session_state:
        st.warning("⚠ Please login first")
    else:
        st.subheader("📜 Detection History")
        if supabase:
            try:
                resp = supabase.table("detection_records").select("*").eq("farmer_username", st.session_state["user"]).execute()
                if resp.data:
                    for rec in resp.data:
                        st.write(f"🗓 {rec['timestamp']} → {rec['prediction']} ({rec['confidence']})")
                else:
                    st.info("No history found.")
            except Exception as e:
                st.error(f"History error: {e}")
        else:
            st.warning("⚠ Supabase not available")
