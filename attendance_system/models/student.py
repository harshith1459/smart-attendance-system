from extensions import db

class Student(db.Document):
    user = db.ReferenceField('User', unique=True, required=True)
    name = db.StringField(max_length=150, required=True)
    roll_no = db.StringField(max_length=50, unique=True, required=True)
    branch = db.StringField(max_length=100)
    face_data_path = db.StringField(max_length=255)
    is_face_enrolled = db.BooleanField(default=False)
    face_enrollment_data = db.ListField(db.StringField()) # Store Base64 frames
    
    meta = {
        'collection': 'students',
        'indexes': ['roll_no', 'user']
    }
