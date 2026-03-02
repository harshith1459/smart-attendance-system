import sys
import os
sys.path.append(os.getcwd())
from mongoengine import connect
from models.user import User
from werkzeug.security import generate_password_hash

# Connect to DB (matching app.py config)
connect(host='mongodb://localhost:27017/attendance_db')

def create_admin():
    username = 'admin'
    password = 'admin_security_pass' # Provide this to user
    
    # Check if exists
    existing = User.objects(username=username).first()
    if existing:
        print(f"User {username} already exists. Updating password...")
        existing.password_hash = generate_password_hash(password)
        existing.role = 'admin'
        existing.is_approved = True
        existing.save()
    else:
        User(
            username=username,
            password_hash=generate_password_hash(password),
            role='admin',
            is_approved=True
        ).save()
        print(f"Admin created: {username} / {password}")

if __name__ == '__main__':
    create_admin()
