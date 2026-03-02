import sys
import os
sys.path.append(os.getcwd())
from mongoengine import connect
from models.user import User
from models.teacher import Teacher
from werkzeug.security import generate_password_hash

# Connect to DB
connect(host='mongodb://localhost:27017/attendance_db')

def create_teacher():
    username = 'vengi'
    password = 'teacher_pass_123'
    
    # Check if exists
    user = User.objects(username=username).first()
    if user:
        print(f"User {username} already exists. Resetting password...")
        user.password_hash = generate_password_hash(password)
        user.role = 'teacher'
        user.is_approved = True
        user.save()
    else:
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role='teacher',
            is_approved=True
        ).save()
        
    # Ensure profile exists
    teacher = Teacher.objects(user=user).first()
    if not teacher:
        Teacher(
            user=user,
            name='Vengi',
            department='Computer Science'
        ).save()
        print(f"Teacher created: {username} / {password}")
    else:
        print(f"Teacher profile verified for {username}")

if __name__ == '__main__':
    create_teacher()
