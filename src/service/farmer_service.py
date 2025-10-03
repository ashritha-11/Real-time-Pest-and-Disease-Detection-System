from dao.farmer_dao import insert_farmer, get_farmer_by_auth

def register_farmer(auth_id, name, email, farm_location, crop_type):
    return insert_farmer(auth_id, name, email, farm_location, crop_type)

def get_farmer(auth_id):
    return get_farmer_by_auth(auth_id)
