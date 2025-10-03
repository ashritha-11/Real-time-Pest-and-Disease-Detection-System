import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    data = supabase.table("detection_records").select("*").limit(1).execute()
    st.success("✅ Connected!")
    st.write(data.data)
except Exception as e:
    st.error(f"Connection failed: {e}")
