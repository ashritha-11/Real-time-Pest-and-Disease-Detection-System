from supabase import create_client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Connected to Supabase!")
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    supabase = None
