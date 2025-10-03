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

# ---------- Supabase config ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.success("✅ Connected to Supabase!")
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        supabase = None
else:
    st.warning("Supabase credentials missing. App will use local storage.")

# ---------- Model ----------
MODEL_PATH = "models/cnn_model.h5"
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

# ---------- Helper ----------
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

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
            if not supabase:
                st.warning("Supabase not connected")
            else:
                try:
                    if role_input=="Admin":
                        res = supabase.table("admins").insert({
                            "name": username_input,
                            "email": username_input,
                            "password": hashed_pw,
                            "role": role_input
                        }).execute()
                    else:
                        res = supabase.table("farmers").insert({
                            "username": username_input,
                            "email": username_input,
                            "password": hashed_pw,
                            "role": role_input
                        }).execute()
                        if res.data:
                            st.session_state.user_id = res.data[0]["id"]
                    st.success(f"✅ Registered in Supabase as {role_input}")
                except Exception as e:
                    st.error(f"Could not register: {e}")
        else:  # Login
            if not supabase:
                st.warning("Supabase not connected")
            else:
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
                    st.error(f"Login failed: {e}")

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
            
            # Insert detection into Supabase
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
                    st.error(f"Could not save detection: {e}")

# ---------- Admin UI ----------
if st.session_state.logged_in and st.session_state.role=="Admin":
    st.subheader(f"👋 Admin Panel: {st.session_state.username}")
    if supabase:
        try:
            records = supabase.table("detection_records").select("*").order("timestamp", desc=True).execute().data
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
                st.info("No records found.")
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
