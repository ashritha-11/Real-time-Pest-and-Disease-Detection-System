class User:
    def __init__(self, user_id, name, email, password):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.password = password

class Farmer(User):
    def __init__(self, user_id, name, email, password, farm_location, crop_type):
        super().__init__(user_id, name, email, password)
        self.farm_location = farm_location
        self.crop_type = crop_type

class Admin(User):
    def __init__(self, user_id, name, email, password, role):
        super().__init__(user_id, name, email, password)
        self.role = role

class DetectionRecord:
    def __init__(self, detection_id, farmer_id, class_id, device_id, image_path, confidence, timestamp):
        self.detection_id = detection_id
        self.farmer_id = farmer_id
        self.class_id = class_id
        self.device_id = device_id
        self.image_path = image_path
        self.confidence = confidence
        self.timestamp = timestamp
