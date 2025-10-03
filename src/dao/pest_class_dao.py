from .supabase_client import supabase

def add_pest_class(name, description):
    return supabase.table("pest_disease_classes").insert({
        "name": name,
        "description": description
    }).execute()

def get_all_pest_classes():
    return supabase.table("pest_disease_classes").select("*").execute()
