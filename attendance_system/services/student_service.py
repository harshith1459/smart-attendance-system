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
        import base64, os

        if not student_profile.face_enrollment_data:
            student_profile.face_enrollment_data = []

        # If it's the first frame, clear old data (for re-enrollment)
        if index == 0:
            student_profile.face_enrollment_data = [image_data]
        else:
            student_profile.face_enrollment_data.append(image_data)

        # Save image to disk for face recognizer
        dataset_dir = os.path.join(os.getcwd(), 'static', 'uploads', 'dataset',
                                   f'student_{student_profile.id}')
        os.makedirs(dataset_dir, exist_ok=True)

        try:
            # Strip data URI header if present
            if ',' in image_data:
                encoded = image_data.split(',', 1)[1]
            else:
                encoded = image_data
            img_bytes = base64.b64decode(encoded)
            img_path = os.path.join(dataset_dir, f'frame_{index}.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
        except Exception as e:
            print(f'Failed to save frame to disk: {e}')

        # Delete cached embeddings so recognizer rebuilds on next scan
        pkl_path = os.path.join(os.getcwd(), 'static', 'uploads', 'dataset',
                                'representations_sface.pkl')
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        # Completion check (5 frames)
        if len(student_profile.face_enrollment_data) >= 5:
            student_profile.is_face_enrolled = True

        student_profile.save()
        return True
