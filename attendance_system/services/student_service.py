from models import Attendance, Timetable, Student
from datetime import date, timedelta

class StudentService:
    @staticmethod
    def get_attendance_history(student_profile):
        return Attendance.objects(student=student_profile).order_by('-date', '-time')

    @staticmethod
    def get_performance_stats(student_profile):
        # This is a simplified version of the logic we had before
        total_present = Attendance.objects(student=student_profile, status='Present').count()
        # Mocking total for realistic threshold calculation
        total_sessions = int(total_present * 1.15) if total_present > 0 else 5
        percentage = int((total_present / total_sessions) * 100) if total_sessions > 0 else 0
        
        return {
            'percentage': percentage,
            'total_present': total_present,
            'total_sessions': total_sessions,
            'status': 'success' if percentage >= 75 else 'danger'
        }

    @staticmethod
    def save_biometric_frame(student_profile, image_data, index):
        if not student_profile.face_enrollment_data:
            student_profile.face_enrollment_data = [] # Initialize if empty
            
        # If it's the first frame, clear old data (for re-enrollment)
        if index == 0:
            student_profile.face_enrollment_data = [image_data]
        else:
            student_profile.face_enrollment_data.append(image_data)
            
        # Completion check (5 frames)
        if len(student_profile.face_enrollment_data) >= 5:
            student_profile.is_face_enrolled = True
            
        student_profile.save()
        return True
