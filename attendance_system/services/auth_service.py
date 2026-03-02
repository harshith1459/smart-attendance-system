from models import User, Student, Teacher
from werkzeug.security import generate_password_hash, check_password_hash

class AuthService:
    @staticmethod
    def authenticate(username, password):
        user = User.objects(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            if not user.is_approved:
                return None, "Awaiting administrative approval."
            return user, None
        return None, "Invalid credentials."

    @staticmethod
    def register_user(form_data):
        hashed_pw = generate_password_hash(form_data['password'])
        
        user = User(
            username=form_data['username'],
            password_hash=hashed_pw,
            role=form_data['role'],
            is_approved=False
        ).save()
        
        if form_data['role'] == 'student':
            Student(
                user=user,
                name=form_data['full_name'],
                roll_no=form_data['roll_no'],
                branch="General"
            ).save()
        elif form_data['role'] == 'teacher':
            Teacher(
                user=user,
                name=form_data['full_name'],
                department=form_data['department']
            ).save()
            
        return user
