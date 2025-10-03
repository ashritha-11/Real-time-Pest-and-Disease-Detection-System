from supabase_client import supabase
import datetime

def add_detection(farmer_id, class_id, device_id, image_url, confidence):
    return supabase.table("detections").insert({
        "farmer_id": farmer_id,
        "class_id": class_id,
        "device_id": device_id,
        "image_url": image_url,
        "confidence": confidence,
        "timestamp": datetime.datetime.now().isoformat()
    }).execute()

def get_detections_for_farmer(farmer_id):
    return supabase.table("detections").select("*").eq("farmer_id", farmer_id).execute()
