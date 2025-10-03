from supabase import create_client, Client
import streamlit as st

# Fetch secrets from Streamlit or environment
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Initialize client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
