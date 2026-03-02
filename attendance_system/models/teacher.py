from extensions import db

class Teacher(db.Document):
    user = db.ReferenceField('User', unique=True, required=True)
    name = db.StringField(max_length=150, required=True)
    department = db.StringField(max_length=100)

    meta = {'collection': 'teachers'}
