from .supabase_client import supabase

def insert_admin(auth_id, name, email, role="admin"):
    return supabase.table("admins").insert({
        "auth_id": auth_id,
        "name": name,
        "email": email,
        "role": role
    }).execute()

def get_admin_by_auth(auth_id):
    return supabase.table("admins").select("*").eq("auth_id", auth_id).execute()
