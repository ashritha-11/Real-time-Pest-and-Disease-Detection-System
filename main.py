import streamlit as st
import os
from service.detection_service import predict_and_store
from dao.detection_dao import get_detections_for_farmer, add_detection
import tensorflow as tf
import auth  # make sure you have auth.py with register_user/login_user

st.title("🌱 Real-time Pest & Disease Detection System")

# --- Session state for login ---
if "user" not in st.session_state:
    st.session_state.user = None

menu = ["Home", "Login", "Register", "Upload Image", "History"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Home Page ---
if choice == "Home":
    st.subheader("Welcome to the AI Pest & Disease Detector 👨‍🌾")
    st.write("Register or login to start detecting crop issues.")

# --- Register Page ---
elif choice == "Register":
    st.subheader("📝 Create an Account")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["farmer", "admin"])

    if st.button("Register"):
        success, msg = auth.register_user(email, password, name, role)
        st.success(msg) if success else st.error(msg)

# --- Login Page ---
elif choice == "Login":
    st.subheader("🔑 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        session = auth.login_user(email, password)
        if session:
            st.session_state.user = session.user
            st.success(f"Welcome {st.session_state.user.email}!")
        else:
            st.error("Invalid login credentials.")

# --- Upload Image Page ---
elif choice == "Upload Image":
    if not st.session_state.user:
        st.warning("Please login first.")
    else:
        st.subheader("📤 Upload Crop Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(save_path, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                # Call your service function
                farmer_id = st.session_state.user.id
                device_id = None  # provide device id if available
                prediction, confidence = predict_and_store(farmer_id, device_id, save_path)

                st.success(f"Prediction: {prediction} (Confidence: {confidence})")

# --- Detection History Page ---
elif choice == "History":
    if not st.session_state.user:
        st.warning("Please login first.")
    else:
        st.subheader("📜 Detection History")
        farmer_id = st.session_state.user.id
        history = get_detections_for_farmer(farmer_id)

        if history:
            for record in history:
                st.write(f"📝 {record['timestamp']} - {record['image_url']} - Confidence: {record['confidence']}")
        else:
            st.info("No history found yet.")
