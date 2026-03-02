import sys
import os
sys.path.append(os.getcwd())
from mongoengine import connect
from models.user import User
from models.student import Student
from werkzeug.security import generate_password_hash

# Connect to DB
connect(host='mongodb://localhost:27017/attendance_db')

def create_student():
    username = 'abhi'
    password = 'student_pass_123'
    
    # Check if exists
    user = User.objects(username=username).first()
    if user:
        print(f"User {username} already exists. Resetting password...")
        user.password_hash = generate_password_hash(password)
        user.is_approved = True
        user.save()
    else:
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            role='student',
            is_approved=True
        ).save()
        
    # Ensure profile exists
    student = Student.objects(user=user).first()
    if not student:
        Student(
            user=user,
            name='Abhi Kumar',
            roll_no='21BCE1001',
            branch='Computer Science'
        ).save()
        print(f"Student created: {username} / {password}")
    else:
        print(f"Student profile verified for {username}")

if __name__ == '__main__':
    create_student()
