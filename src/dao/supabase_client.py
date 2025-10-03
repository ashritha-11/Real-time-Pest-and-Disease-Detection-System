import streamlit as st
from supabase import create_client, Client

# Use Streamlit secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client | None = None

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Connected to Supabase!")
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    supabase = None
