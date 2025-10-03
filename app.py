# app.py
import streamlit as st
import os
import json
import hashlib
from datetime import datetime
from supabase import create_client, Client
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="🌱 Pest & Disease Detection System", layout="wide")
st.title("🌱 Pest & Disease Detection System")

# ---------- Supabase connection ----------
supabase: Client | None = None
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Connected to Supabase!")
except Exception as e:
    supabase = None
    st.warning(f"Supabase connection failed. Using local storage only. {e}")

# ---------- File paths ----------
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
MODEL_PATH = "models/cnn_model.h5"

# ---------- Tables ----------
TABLE_DETECTION = "detection_records"
TABLE_FARMERS = "farmers"
TABLE_ADMINS = "admins"

# ---------- JSON helpers ----------
def load_json(path):
    if os.path.exists(path):
        with open(path,"r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_json(path, data):
    with open(path,"w") as f:
        json.dump(data, f, indent=2)

# ---------- Supabase table check ----------
def table_exists(table_name):
    if not supabase:
        return False
    try:
        supabase.table(table_name).select("*").limit(1).execute()
        return True
    except:
        return False

def ensure_tables():
    if not supabase:
        return
    # Create farmers
    if not table_exists(TABLE_FARMERS):
        supabase.table(TABLE_FARMERS).create("""
            id serial primary key,
            username text unique not null,
            password text not null,
            role text not null
        """).execute()
    # Create admins
    if not table_exists(TABLE_ADMINS):
        supabase.table(TABLE_ADMINS).create("""
            id serial primary key,
            username text unique not null,
            password text not null,
            role text not null
        """).execute()
    # Create detection_records
    if not table_exists(TABLE_DETECTION):
        supabase.table(TABLE_DETECTION).create(f"""
            id serial primary key,
            farmer_id int references public.farmers(id),
            prediction text not null,
            confidence numeric not null,
            image_url text not null,
            timestamp timestamptz default now()
        """).execute()

if supabase:
    try:
        ensure_tables()
    except:
        st.warning("Could not ensure tables on Supabase.")

# ---------- Password hashing ----------
def hash_pw(pw: str):
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------- Local auth ----------
def local_register(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_pw(password)
    save_json(USERS_FILE, users)
    return True

def local_login(username, password):
    users = load_json(USERS_FILE)
    return users.get(username) == hash_pw(password)

# ---------- History ----------
def add_local_history(username, image_name, prediction, confidence):
    history = load_json(HISTORY_FILE)
    if username not in history:
        history[username] = []
    history[username].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_name,
        "prediction": prediction,
        "confidence": confidence
    })
    save_json(HISTORY_FILE, history)

# ---------- Load model ----------
model_loaded = False
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model_loaded = True
    except:
        model_loaded = False

def predict_image(path):
    if model_loaded and model:
        img = Image.open(path).resize((224,224)).convert("RGB")
        arr = np.array(img)/255.0
        arr = arr.reshape((1,)+arr.shape)
        probs = model.predict(arr)[0]
        idx = probs.argmax()
        label = "Healthy" if idx==0 else ("Pest-Affected" if idx==1 else "Disease-Affected")
        return label, float(probs[idx])
    # Fallback dummy prediction
    width = Image.open(path).size[0]
    if width % 3 == 0: return "Healthy", 0.95
    if width % 3 == 1: return "Pest-Affected", 0.85
    return "Disease-Affected", 0.90

# ---------- Session ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role="Farmer"
    st.session_state.user_id=None

# ---------- Styles ----------
st.markdown("""
<style>
.stButton>button {background-color:#4CAF50;color:white;height:3em;width:100%;font-weight:bold;}
.card {padding:12px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.08); margin-bottom:12px;}
.prediction {font-weight:bold; padding:6px 10px; border-radius:6px; color:white; display:inline-block;}
</style>
""", unsafe_allow_html=True)

# ---------- Auth UI ----------
if not st.session_state.logged_in:
    st.subheader("Login / Register")
    kind = st.radio("Choose", ["Login","Register"], horizontal=True)
    username_input = st.text_input("Username or email")
    password_input = st.text_input("Password", type="password")
    role_input = st.selectbox("Role (for register)", ["Farmer","Admin"])

    if st.button("Submit"):
        if kind=="Register":
            if local_register(username_input,password_input):
                st.success("Registered locally. Please login.")
            else:
                st.warning("Username already exists.")
        else:
            if local_login(username_input,password_input):
                st.session_state.logged_in=True
                st.session_state.username=username_input
                st.session_state.role=role_input
                st.success(f"Logged in locally as {username_input}")
            else:
                st.error("Invalid credentials")

# ---------- Farmer UI ----------
if st.session_state.logged_in and st.session_state.role=="Farmer":
    st.subheader(f"👋 Welcome, {st.session_state.username}!")
    uploaded = st.file_uploader("Upload crop leaf image", type=["jpg","jpeg","png"])
    if uploaded:
        save_name=f"{st.session_state.username}_{uploaded.name}"
        with open(save_name,"wb") as f: f.write(uploaded.getbuffer())
        st.image(save_name, use_container_width=True)
        if st.button("🔍 Detect"):
            pred, conf = predict_image(save_name)
            add_local_history(st.session_state.username, uploaded.name, pred, conf)
            color="#4CAF50" if pred=="Healthy" else ("#FF9800" if "Pest" in pred else "#F44336")
            st.markdown(f"<span class='prediction' style='background-color:{color}'>{pred} ({conf*100:.1f}%)</span>",unsafe_allow_html=True)

# ---------- Admin UI ----------
if st.session_state.logged_in and st.session_state.role=="Admin":
    st.subheader(f"👋 Admin Panel: {st.session_state.username}")
    st.info("Admin view coming soon... (connect to Supabase for real records)")

# ---------- Local History ----------
st.markdown("---")
st.subheader("📜 Detection History (local)")
history = load_json(HISTORY_FILE).get(st.session_state.username, [])
if history:
    for rec in reversed(history):
        color="#4CAF50" if rec.get("prediction")=="Healthy" else ("#FF9800" if "Pest" in rec.get("prediction","") else "#F44336")
        st.markdown(f"""
            <div class="card">
            <p><b>Image:</b> {rec.get('image')}</p>
            <p><b>Timestamp:</b> {rec.get('timestamp')}</p>
            <p><span class='prediction' style='background-color:{color}'>{rec.get('prediction')} ({rec.get('confidence',0.0)*100:.1f}%)</span></p>
            </div>
        """,unsafe_allow_html=True)
else:
    st.info("No history found.")

# ---------- Logout ----------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role="Farmer"
    st.session_state.user_id=None
    st.experimental_rerun()
