from extensions import db

class Timetable(db.Document):
    teacher = db.ReferenceField('Teacher', required=True)
    subject_name = db.StringField(max_length=150, required=True)
    subject_code = db.StringField(max_length=20)
    day_of_week = db.StringField(max_length=20, required=True)
    period_no = db.IntField()
    start_time = db.StringField(required=True) # "HH:MM"
    end_time = db.StringField(required=True)   # "HH:MM"

    meta = {'collection': 'timetables'}
