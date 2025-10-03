from supabase_client import supabase
from datetime import datetime

# Insert detection into detection_records table
def add_detection(farmer_id, prediction, confidence, image_url):
    try:
        res = supabase.table("detection_records").insert({
            "farmer_id": farmer_id,
            "prediction": prediction,
            "confidence": confidence,
            "image_url": image_url,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        return res
    except Exception as e:
        print("Error inserting detection:", e)
        return None

# Get all detections for a farmer
def get_detections_for_farmer(farmer_id):
    try:
        res = supabase.table("detection_records").select("*").eq("farmer_id", farmer_id).execute()
        return res.data
    except Exception as e:
        print("Error fetching detections:", e)
        return []
