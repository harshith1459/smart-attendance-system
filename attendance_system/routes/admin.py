from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import current_user
from services.admin_service import AdminService
from models import Student, Timetable, Teacher

admin_bp = Blueprint('admin', __name__)

@admin_bp.before_request
def admin_required():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    if current_user.role != 'admin':
        flash('Security Alert: Administrative access required.', 'danger')
        return redirect(url_for('auth.login'))

@admin_bp.route('/dashboard')
def dashboard():
    stats = AdminService.get_dashboard_stats()
    pending_users = AdminService.get_pending_approvals()
    return render_template('admin/dashboard.html', stats=stats, pending_users=pending_users)

@admin_bp.route('/approve/<user_id>')
def approve_user(user_id):
    success, message = AdminService.approve_user(user_id)
    flash(message, 'success' if success else 'danger')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/reject/<user_id>')
def reject_user(user_id):
    success, message = AdminService.reject_user(user_id)
    flash(message, 'info' if success else 'danger')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/students')
def students_list():
    students = Student.objects.all().order_by('roll_no')
    return render_template('admin/students_list.html', students=students)

@admin_bp.route('/student/<student_id>')
def student_details(student_id):
    from services.student_service import StudentService
    student = Student.objects(id=student_id).first()
    if not student:
        abort(404)
    history = StudentService.get_attendance_history(student)
    return render_template('admin/student_details.html', student=student, history=history)

@admin_bp.route('/manage_timetable')
def manage_timetable():
    teachers = Teacher.objects.all()
    grid_data = AdminService.get_timetable_grid()
    timings = {1: ['09:00', '10:00'], 2: ['10:00', '11:00'], 3: ['11:10', '12:10'], 4: ['12:10', '01:10'], 5: ['02:00', '03:00'], 6: ['03:00', '04:00'], 7: ['04:00', '05:00']}
    return render_template('admin/timetable.html', grid_data=grid_data, timings=timings, teachers=teachers)
