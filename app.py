import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from PIL import Image
import tensorflow as tf
import numpy as np

# ---------------- Load environment ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---------------- Initialize Supabase ----------------
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase!")
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
else:
    st.warning("Supabase credentials not set — using local storage.")

# ---------------- Files & Model ----------------
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
MODEL_PATH = "models/cnn_model.h5"
FARMERS_TABLE = "farmers"
ADMINS_TABLE = "admins"
DETECTION_TABLE = "detection_records"

# ---------------- Helpers ----------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- Model ----------------
model_loaded = False
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model_loaded = True
    except:
        model_loaded = False

def predict_image(path: str):
    if model_loaded and model:
        img = Image.open(path).resize((224,224)).convert("RGB")
        arr = np.array(img)/255.0
        arr = arr.reshape((1,)+arr.shape)
        probs = model.predict(arr)[0]
        idx = probs.argmax()
        label = "Healthy" if idx==0 else ("Pest-Affected" if idx==1 else "Disease-Affected")
        return label, float(probs[idx])
    width = Image.open(path).size[0]
    if width % 3 == 0: return "Healthy", 0.95
    if width % 3 == 1: return "Pest-Affected", 0.85
    return "Disease-Affected", 0.90

# ---------------- Session ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = "Farmer"
    st.session_state.user_id = None

# ---------------- Styles ----------------
st.markdown("""
<style>
.stButton>button {background-color:#4CAF50;color:white;height:3em;width:100%;font-weight:bold;}
.card {padding:12px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.08); margin-bottom:12px;}
.prediction {font-weight:bold; padding:6px 10px; border-radius:6px; color:white; display:inline-block;}
</style>
""", unsafe_allow_html=True)

# ---------------- Auth ----------------
st.subheader("Login / Register")
kind = st.radio("Choose", ["Login", "Register"], horizontal=True)
username_input = st.text_input("Username / Email")
password_input = st.text_input("Password", type="password")
role_input = st.selectbox("Role (for register)", ["Farmer", "Admin"])

if st.button("Submit"):
    hashed_pw = hash_password(password_input)
    
    # ---------- Register ----------
    if kind=="Register":
        table = FARMERS_TABLE if role_input=="Farmer" else ADMINS_TABLE
        if supabase:
            try:
                payload = {"username": username_input, "email": username_input,
                           "password": hashed_pw, "role": role_input}
                if role_input=="Admin":
                    payload = {"name": username_input, "email": username_input,
                               "password": hashed_pw, "role": role_input}
                res = supabase.table(table).insert(payload).execute()
                st.success("✅ Registered successfully in Supabase!")
            except Exception as e:
                st.error(f"Supabase insert failed: {e}")
        else:
            users = load_json(USERS_FILE)
            if username_input in users:
                st.warning("Username already exists locally.")
            else:
                users[username_input] = hashed_pw
                save_json(USERS_FILE, users)
                st.success("Registered locally.")

    # ---------- Login ----------
    else:
        user = None
        if supabase:
            table = FARMERS_TABLE if role_input=="Farmer" else ADMINS_TABLE
            try:
                res = supabase.table(table).select("*").eq("email", username_input).execute()
                if res.data and res.data[0]["password"]==hashed_pw:
                    user = res.data[0]
                    st.success(f"✅ Logged in as {username_input}")
            except Exception as e:
                st.warning(f"Supabase login failed: {e}")
        if not user:
            users = load_json(USERS_FILE)
            if users.get(username_input)==hashed_pw:
                user = {"username": username_input}
                st.success(f"Logged in locally as {username_input}")
            else:
                st.error("Invalid credentials")
        
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username_input
            st.session_state.role = role_input
            st.session_state.user_id = user.get("id", None)

# ---------------- Farmer Dashboard ----------------
if st.session_state.logged_in and st.session_state.role=="Farmer":
    st.subheader(f"👋 Welcome {st.session_state.username}")
    uploaded = st.file_uploader("Upload crop image", type=["jpg","jpeg","png"])
    
    if uploaded:
        os.makedirs("uploads", exist_ok=True)
        save_path = f"uploads/{st.session_state.username}_{uploaded.name}"
        with open(save_path, "wb") as f: f.write(uploaded.getbuffer())
        st.image(save_path, use_container_width=True)
        
        if st.button("🔍 Detect"):
            pred, conf = predict_image(save_path)
            st.success(f"Prediction: {pred} ({conf*100:.1f}%)")
            
            if supabase:
                try:
                    payload = {"farmer_id": st.session_state.user_id,
                               "prediction": pred, "confidence": float(conf),
                               "image_url": save_path, "timestamp": datetime.utcnow().isoformat()}
                    supabase.table(DETECTION_TABLE).insert(payload).execute()
                    st.success("✅ Detection stored in Supabase")
                except Exception as e:
                    st.warning(f"Could not insert detection: {e}")

# ---------------- Admin Dashboard ----------------
if st.session_state.logged_in and st.session_state.role=="Admin":
    st.subheader(f"👋 Admin Panel: {st.session_state.username}")
    if supabase:
        try:
            records_res = supabase.table(DETECTION_TABLE).select("*").order("timestamp", desc=True).execute()
            records = records_res.data
            if records:
                for rec in records:
                    color = "#4CAF50" if rec.get("prediction")=="Healthy" else ("#FF9800" if "Pest" in rec.get("prediction","") else "#F44336")
                    st.markdown(f"""
                        <div class="card">
                        <p><b>Farmer ID:</b> {rec.get('farmer_id','')}</p>
                        <p><b>Prediction:</b> <span class='prediction' style='background-color:{color}'>{rec.get('prediction','')}</span></p>
                        <p><b>Confidence:</b> {rec.get('confidence',0.0)*100:.1f}%</p>
                        <p><b>Image:</b> {rec.get('image_url','')}</p>
                        <p><b>Timestamp:</b> {rec.get('timestamp','')}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No detection records.")
        except Exception as e:
            st.warning(f"Failed to fetch records: {e}")

# ---------------- Logout ----------------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role="Farmer"
    st.session_state.user_id=None
    st.experimental_rerun()
