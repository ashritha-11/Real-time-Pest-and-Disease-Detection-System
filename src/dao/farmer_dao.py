from .supabase_client import supabase

def insert_farmer(auth_id, name, email, farm_location, crop_type):
    data = {
        "auth_id": auth_id,
        "name": name,
        "email": email,
        "farm_location": farm_location,
        "crop_type": crop_type
    }
    return supabase.table("farmers").insert(data).execute()

def get_farmer_by_auth(auth_id):
    return supabase.table("farmers").select("*").eq("auth_id", auth_id).execute()
