from supabase_client import supabase
import hashlib

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(email, password, name, role):
    table = "admins" if role=="admin" else "farmers"
    hashed_pw = hash_password(password)
    try:
        res = supabase.table(table).insert({
            "username": name,
            "email": email,
            "password": hashed_pw,
            "role": role
        }).execute()
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Error: {e}"

def login_user(email, password, role):
    table = "admins" if role=="admin" else "farmers"
    hashed_pw = hash_password(password)
    try:
        res = supabase.table(table).select("*").eq("email", email).execute()
        if res.data and res.data[0]["password"] == hashed_pw:
            return res.data[0]
        return None
    except Exception as e:
        print("Login error:", e)
        return None
