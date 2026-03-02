from models import Attendance, Timetable, Student
from datetime import date, datetime

class AttendanceService:
    @staticmethod
    def get_today_stats(teacher_profile):
        today_name = datetime.now().strftime('%A')
        today_classes = Timetable.objects(teacher=teacher_profile, day_of_week=today_name).order_by('period_no')
        
        class_stats = []
        for cls in today_classes:
            count = Attendance.objects(
                teacher=teacher_profile,
                course=cls,
                date=date.today(),
                status='Present'
            ).count()
            
            class_stats.append({
                'period': cls.period_no,
                'subject': cls.subject_name,
                'time': f"{cls.start_time} - {cls.end_time}",
                'present_count': count
            })
            
        total_marked = Attendance.objects(
            teacher=teacher_profile,
            date=date.today(),
            status='Present'
        ).count()
            
        return class_stats, total_marked

    @staticmethod
    def get_current_slot(teacher_profile):
        now = datetime.now()
        day_name = now.strftime('%A')
        current_time_str = now.strftime('%H:%M')
        
        return Timetable.objects(
            teacher=teacher_profile,
            day_of_week=day_name,
            start_time__lte=current_time_str,
            end_time__gte=current_time_str
        ).first()

    @staticmethod
    def record_attendance(student, teacher_profile, course=None, method='Auto-ML'):
        # Check duplication
        if course:
            exists = Attendance.objects(student=student, date=date.today(), course=course).first()
        else:
            exists = Attendance.objects(student=student, date=date.today(), teacher=teacher_profile).first()
            
        if not exists:
            Attendance(
                student=student,
                teacher=teacher_profile,
                course=course,
                date=date.today(),
                time=datetime.now(),
                status='Present',
                method=method
            ).save()
            return True
        return False
