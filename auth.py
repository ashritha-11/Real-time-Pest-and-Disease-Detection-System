# auth.py
USERS = {}

def register_user(email, password, name, role):
    if email in USERS:
        return False, "User already exists!"
    USERS[email] = {"password": password, "name": name, "role": role, "id": len(USERS)+1}
    return True, "Registration successful!"

def login_user(email, password):
    user = USERS.get(email)
    if user and user["password"] == password:
        return type("Session", (), {"user": user})()  # return simple session object
    return None
