from extensions import db
from flask_login import UserMixin

class User(UserMixin, db.Document):
    username = db.StringField(max_length=150, unique=True, required=True)
    password_hash = db.StringField(max_length=256, required=True)
    role = db.StringField(max_length=50, required=True)  # 'admin', 'teacher', 'student'
    is_approved = db.BooleanField(default=True)
    
    meta = {'allow_inheritance': True, 'collection': 'users'}

    def get_id(self):
        return str(self.id)

    @property
    def student_profile(self):
        from .student import Student
        return Student.objects(user=self).first()

    @property
    def teacher_profile(self):
        from .teacher import Teacher
        return Teacher.objects(user=self).first()
