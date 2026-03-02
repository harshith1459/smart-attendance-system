from extensions import db
from datetime import datetime

class Attendance(db.Document):
    student = db.ReferenceField('Student', required=True)
    course = db.ReferenceField('Timetable')
    teacher = db.ReferenceField('Teacher', required=True)
    date = db.DateField(default=datetime.now)
    time = db.DateTimeField(default=datetime.now)
    status = db.StringField(max_length=20, default='Present')
    method = db.StringField(max_length=50) # 'Auto-ML', 'Manual'
    
    meta = {
        'collection': 'attendance',
        'indexes': [
            ('student', 'date'),
            ('date', 'course'),
            '-date'
        ]
    }

class AttendanceRequest(db.Document):
    teacher = db.ReferenceField('Teacher', required=True)
    student = db.ReferenceField('Student', required=True)
    date = db.DateField(required=True)
    period_no = db.IntField()
    reason = db.StringField(required=True)
    status = db.StringField(max_length=20, default='Pending')

    meta = {'collection': 'attendance_requests'}
