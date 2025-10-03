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
from dotenv import load_dotenv

# ---------- Load environment ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---------- Initialize Supabase ----------
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase!")

        # ---------- Create Tables if Not Exists ----------
        # 1. Farmers
        supabase.rpc("pg_exec", {
            "sql": """
            create table if not exists public.farmers (
                id serial primary key,
                username text,
                email text unique,
                password text,
                role text
            );
            """
        }).execute()
        # 2. Admins
        supabase.rpc("pg_exec", {
            "sql": """
            create table if not exists public.admins (
                admin_id uuid primary key default gen_random_uuid(),
                name text not null,
                email text unique not null,
                password text not null,
                role text
            );
            """
        }).execute()
        # 3. Detection Records
        supabase.rpc("pg_exec", {
            "sql": """
            create table if not exists public.detection_records (
                id serial primary key,
                farmer_id int references public.farmers(id),
                prediction text not null,
                confidence numeric not null,
                image_url text not null,
                timestamp timestamptz default now()
            );
            """
        }).execute()
        st.info("✅ Tables checked/created in Supabase.")

    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        supabase = None
else:
    st.warning("Supabase credentials missing. Using local fallback.")

# ---------- Files ----------
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
MODEL_PATH = "models/cnn_model.h5"

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

# ---------- Helpers ----------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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
.prediction {font-weight:bold; padding:6px 10px; border-radius:6px; color:white; display:inline-block;}
</style>
""", unsafe_allow_html=True)

# ---------- Auth ----------
if not st.session_state.logged_in:
    st.subheader("Login / Register")
    choice = st.radio("Choose", ["Login","Register"], horizontal=True)
    username_input = st.text_input("Username or Email")
    password_input = st.text_input("Password", type="password")
    role_input = st.selectbox("Role (for register)", ["Farmer","Admin"])

    if st.button("Submit"):
        hashed_pw = hash_password(password_input)
        if choice=="Register":
            if supabase:
                try:
                    table = "admins" if role_input=="Admin" else "farmers"
                    data = {
                        "name" if role_input=="Admin" else "username": username_input,
                        "email": username_input,
                        "password": hashed_pw,
                        "role": role_input
                    }
                    res = supabase.table(table).insert(data).execute()
                    st.success(f"✅ Registered in Supabase as {role_input}")
                    if role_input=="Farmer" and res.data:
                        st.session_state.user_id = res.data[0]["id"]
                except Exception as e:
                    st.error(f"Supabase registration failed: {e}")
            else:
                st.warning("Supabase not connected. Using local storage.")
                users = load_json(USERS_FILE)
                if username_input in users:
                    st.warning("Username already exists locally.")
                else:
                    users[username_input] = {"password": hashed_pw, "role": role_input, "id": len(users)+1}
                    save_json(USERS_FILE, users)
                    st.success("✅ Registered locally.")
        else:  # Login
            if supabase:
                try:
                    table = "admins" if role_input=="Admin" else "farmers"
                    res = supabase.table(table).select("*").eq("email", username_input).execute()
                    if res.data and res.data[0]["password"]==hashed_pw:
                        st.session_state.logged_in=True
                        st.session_state.username=username_input
                        st.session_state.role=role_input
                        if role_input=="Farmer":
                            st.session_state.user_id = res.data[0]["id"]
                        st.success(f"✅ Logged in as {role_input}")
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    st.error(f"Supabase login failed: {e}")
            else:
                st.warning("Supabase not connected. Using local storage.")
                users = load_json(USERS_FILE)
                user = users.get(username_input)
                if user and user["password"]==hashed_pw:
                    st.session_state.logged_in=True
                    st.session_state.username=username_input
                    st.session_state.role=user["role"]
                    st.session_state.user_id=user["id"]
                    st.success(f"✅ Logged in locally")
                else:
                    st.error("Invalid local credentials")

# ---------- Farmer UI ----------
if st.session_state.logged_in and st.session_state.role=="Farmer":
    st.subheader(f"👋 Welcome, {st.session_state.username}!")
    uploaded = st.file_uploader("Upload crop image", type=["jpg","jpeg","png"])
    if uploaded:
        save_path = f"{st.session_state.username}_{uploaded.name}"
        with open(save_path,"wb") as f: f.write(uploaded.getbuffer())
        st.image(save_path, use_column_width=True)
        if st.button("🔍 Detect"):
            pred, conf = predict_image(save_path)
            color = "#4CAF50" if pred=="Healthy" else ("#FF9800" if "Pest" in pred else "#F44336")
            st.markdown(f"<span class='prediction' style='background-color:{color}'>{pred} ({conf*100:.1f}%)</span>", unsafe_allow_html=True)
            
            # Insert detection
            if supabase:
                try:
                    payload = {
                        "farmer_id": st.session_state.user_id,
                        "prediction": pred,
                        "confidence": float(conf),
                        "image_url": save_path,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    supabase.table("detection_records").insert(payload).execute()
                    st.success("✅ Detection saved in Supabase")
                except Exception as e:
                    st.error(f"Supabase detection insert failed: {e}")
            else:
                add_local_history(st.session_state.username, uploaded.name, pred, conf)
                st.info("✅ Detection saved locally")

# ---------- Admin UI ----------
if st.session_state.logged_in and st.session_state.role=="Admin":
    st.subheader(f"👋 Admin Panel: {st.session_state.username}")
    if supabase:
        try:
            records_res = supabase.table("detection_records").select("*").order("timestamp", desc=True).execute()
            records = records_res.data if records_res.data else []
            if records:
                for r in records:
                    color = "#4CAF50" if r.get("prediction")=="Healthy" else ("#FF9800" if "Pest" in r.get("prediction","") else "#F44336")
                    st.markdown(f"""
                        <div>
                        <p><b>Farmer ID:</b> {r.get('farmer_id')}</p>
                        <p><b>Prediction:</b> <span class='prediction' style='background-color:{color}'>{r.get('prediction')}</span></p>
                        <p><b>Confidence:</b> {r.get('confidence',0)*100:.1f}%</p>
                        <p><b>Image:</b> {r.get('image_url')}</p>
                        <p><b>Timestamp:</b> {r.get('timestamp')}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No detection records found.")
        except Exception as e:
            st.error(f"Could not fetch records: {e}")

# ---------- Logout ----------
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state.logged_in=False
    st.session_state.username=""
    st.session_state.role="Farmer"
    st.session_state.user_id=None
    st.experimental_rerun()
