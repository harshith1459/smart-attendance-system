from models import Student, Teacher, Attendance, AttendanceRequest, User
from datetime import date

class AdminService:
    @staticmethod
    def get_dashboard_stats():
        return {
            'teachers': Teacher.objects.count(),
            'students': Student.objects.count(),
            'attendance': Attendance.objects.count(),
            'pending_requests': AttendanceRequest.objects(status='Pending').count()
        }

    @staticmethod
    def get_pending_approvals():
        return User.objects(is_approved=False).order_by('role')

    @staticmethod
    def approve_user(user_id):
        user = User.objects(id=user_id).first()
        if user:
            user.is_approved = True
            user.save()
            return True, f"User {user.username} approved successfully."
        return False, "User not found."

    @staticmethod
    def reject_user(user_id):
        user = User.objects(id=user_id).first()
        if not user:
            return False, "User not found."
        
        username = user.username
        if user.role == 'teacher':
            Teacher.objects(user=user).delete()
        elif user.role == 'student':
            Student.objects(user=user).delete()
        user.delete()
        return True, f"Registration for {username} rejected and account removed."

    @staticmethod
    def get_timetable_grid():
        from models import Timetable
        timetables = Timetable.objects.all()
        grid = {} # {Day: {Period: Entry}}
        for tt in timetables:
            if tt.day_of_week not in grid:
                grid[tt.day_of_week] = {}
            grid[tt.day_of_week][tt.period_no] = tt
        return grid
