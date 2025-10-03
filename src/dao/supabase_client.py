from supabase import Client, create_client
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Insert a record
data = {
    "user_id": 1,
    "image_name": "temp.jpg",
    "prediction": "Pest_Affected"
}
supabase.table("detections").insert(data).execute()  # Works in 0.x versions
