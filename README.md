🌱 Real-time Pest and Disease Detection System

This project is a Streamlit-based web application integrated with Supabase for authentication and storage. It allows farmers to upload crop images, get real-time predictions of whether the plant is Healthy or Diseased, and keeps detection history records. Admins can manage users and view detection reports.

🚀 Features

👨‍🌾 Farmer Features

Register/Login securely

Upload crop images for analysis

Get real-time predictions (Healthy / Diseased / Not Healthy)

View detection history

👩‍💼 Admin Features

Register/Login as Admin

View all detection records across farmers

Manage farmer accounts (monitor activity)

🔗 Backend & Storage

Supabase PostgreSQL used for user authentication & history storage

Uploaded images stored in Supabase Storage

Predictions saved in detection records with timestamp

🛠️ Tech Stack

Frontend: Streamlit

Backend: Supabase (PostgreSQL + Authentication + Storage)

ML Model: Trained PyTorch/TensorFlow model for pest/disease classification

Database Tables:

farmers – stores farmer login details

admins – stores admin login details

detection_records – stores predictions, confidence score, and image URLs
