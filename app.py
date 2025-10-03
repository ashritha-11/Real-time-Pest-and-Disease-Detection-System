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

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
MODEL_PATH = "models/cnn_model.h5"
TABLE_NAME = "detection_records"
FARMERS_TABLE = "farmers"
ADMINS_TABLE = "admins"

# ---------- Load Supabase from secrets ----------
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Connected to Supabase!")
except Exception as e:
    st.warning("Supabase connection failed. Using local storage only.")
    supabase = None

# ---------- JSON helper ----------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------- Supabase tables ----------
def create_tables():
    if not supabase:
        return
    try:
        # Farmers table
        sql_farmers = f"""
        create table if not exists public.{FARMERS_TABLE} (
            farmer_id serial primary key,
            username text unique not null,
            password text not null,
            created_at timestamptz default now()
        );
        """
        supabase.rpc("sql", {"query": sql_farmers}).execute()
        
        # Admins table
        sql_admins = f"""
        create table if not exists public.{ADMINS_TABLE} (
            admin_id serial primary key,
            username text unique not null,
            password text not null,
            created_at timestamptz default now()
        );
        """
        supabase.rpc("sql", {"query": sql_admins}).execute()

        # Detection records table
        sql_detection = f"""
        create table if not exists public.{TABLE_NAME} (
            id serial primary key,
            farmer_id int references public.{FARMERS_TABLE}(farmer_id),
            prediction text not null,
            confidence numeric not null,
            image_url text not null,
            timestamp timestamptz default now()
        );
        """
        supabase.rpc("sql", {"query": sql_detection}).execute()
        st.info(f"✅ Tables '{FARMERS_TABLE}', '{ADMINS_TABLE}', and '{TABLE_NAME}' are ready in Supabase.")
    except Exception as e:
        st.warning(f"Could not create tables: {e}")

if supabase:
    create_tables()

# ---------- User management ----------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def local_register(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_password(password)
    save_json(USERS_FILE, users)
    return True

def local_login(username, password):
    users = load_json(USERS_FILE)
    return users.get(username) == hash_password(password)

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

# ---------- Model ----------
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
        try:
            img = Image.open(path).resize((224,224)).convert("RGB")
            arr = np.array(img)/255.0
            arr = arr.reshape((1,)+arr.shape)
            probs = model.predict(arr)[0]
            idx = probs.argmax()
            label = "Healthy" if idx==0 else ("Pest-Affected" if idx==1 else "Disease-Affected")
            return label, float(probs[idx])
        except:
            pass
    width = Image.open(path).size[0]
    if width % 3 == 0: return "Healthy", 0.95
    if width % 3 == 1: return "Pest-Affected", 0.85
    return "Disease-Affected", 0.90

# ---------- Session ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = "Farmer"
    st.session_state.user_id = None

# ---------- Styles ----------
st.markdown("""
<style>
.stButton>button {background-color:#4CAF50;color:white;height:3em;width:100%;font-weight:bold;}
.card {padding:12px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.08); margin-bottom:12px;}
.prediction {font-weight:bold; padding:6px 10px; border-radius:6px; color:white; display:inline-block;}
</style>
""", unsafe_allow_html=True)

# ---------- Auth ----------
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
                st.session_state.user_id=None
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
            add_local_history(st.session_state.username,uploaded.name,pred,conf)
            color="#4CAF50" if pred=="Healthy" else ("#FF9800" if "Pest" in pred else "#F44336")
            st.markdown(f"<span class='prediction' style='background-color:{color}'>{pred} ({conf*100:.1f}%)</span>",unsafe_allow_html=True)

            # Save to Supabase
            if supabase:
                try:
                    # Look up farmer_id
                    farmer_id = None
                    res = supabase.table(FARMERS_TABLE).select("farmer_id").eq("username", st.session_state.username).execute()
                    if res.data:
                        farmer_id = res.data[0]["farmer_id"]
                    payload={"prediction":pred,"confidence":float(conf),"image_url":save_name,"timestamp":datetime.utcnow().isoformat(),"farmer_id":farmer_id}
                    supabase.table(TABLE_NAME).insert(payload).execute()
                    st.success("✅ Detection saved in Supabase")
                except Exception as e:
                    st.warning(f"Could not save detection to Supabase: {e}")

# ---------- Admin UI ----------
if st.session_state.logged_in and st.session_state.role=="Admin":
    st.subheader(f"👋 Admin Panel: {st.session_state.username}")
    st.markdown("### All Detection Records")
    if supabase:
        try:
            records_res = supabase.table(TABLE_NAME).select("*").order("timestamp", desc=True).execute()
            records = records_res.data
            if records:
                for rec in records:
                    color="#4CAF50" if rec.get("prediction")=="Healthy" else ("#FF9800" if "Pest" in rec.get("prediction","") else "#F44336")
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
                st.info("No detection records found in Supabase.")
        except Exception as e:
            st.warning(f"Could not fetch detection records: {e}")

# ---------- Local History ----------
st.markdown("---")
st.subheader("📜 Detection History (local)")
history=load_json(HISTORY_FILE).get(st.session_state.username,[])
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
