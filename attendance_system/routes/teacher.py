from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from services.attendance_service import AttendanceService
from models import Student, Timetable, Attendance, AttendanceRequest
from forms import AttendanceRequestForm
from datetime import datetime, date
import random

teacher_bp = Blueprint('teacher', __name__)

@teacher_bp.before_request
def teacher_required():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    if current_user.role != 'teacher':
        flash('Security Alert: Teacher credentials required.', 'danger')
        return redirect(url_for('auth.login'))

@teacher_bp.route('/dashboard')
def dashboard():
    teacher_profile = current_user.teacher_profile
    class_stats, total_marked = AttendanceService.get_today_stats(teacher_profile)
    return render_template('teacher/dashboard.html', 
                           class_stats=class_stats, 
                           total_records=total_marked)

@teacher_bp.route('/mark_attendance', methods=['GET', 'POST'])
def mark_attendance():
    if request.method == 'POST':
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No identity packet received.'})
            
        teacher_profile = current_user.teacher_profile
        current_course = AttendanceService.get_current_slot(teacher_profile)
        
        # Simulated Face Detection for Prototyping
        all_students = Student.objects(is_face_enrolled=True)
        detected_rolls = []
        
        if all_students:
            # Simulate 1-3 detections
            picked = random.sample(list(all_students), min(len(all_students), random.randint(1, 3)))
            for s in picked:
                if AttendanceService.record_attendance(s, teacher_profile, current_course):
                    detected_rolls.append(s.roll_no)
        
        return jsonify({
            'status': 'success',
            'total_detected': len(detected_rolls),
            'names': detected_rolls,
            'subject': current_course.subject_name if current_course else "General Session"
        })

    return render_template('teacher/attendance.html', ml_available=True)

@teacher_bp.route('/get_periods')
def get_periods():
    """API: return teacher's periods for a given date (used by request.html & bulk_attendance.html)"""
    date_str = request.args.get('date')
    if not date_str:
        return jsonify([])
    try:
        selected_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify([])

    day_name = selected_date.strftime('%A')
    teacher_profile = current_user.teacher_profile
    slots = Timetable.objects(teacher=teacher_profile, day_of_week=day_name).order_by('period_no')

    result = []
    for slot in slots:
        result.append({
            'period_no': slot.period_no,
            'display': f"Period {slot.period_no} — {slot.subject_name} ({slot.start_time}-{slot.end_time})"
        })
    return jsonify(result)

@teacher_bp.route('/bulk_attendance', methods=['GET', 'POST'])
def bulk_attendance():
    """Manual bulk attendance marking"""
    teacher_profile = current_user.teacher_profile
    students = Student.objects.all().order_by('roll_no')

    if request.method == 'POST':
        date_str = request.form.get('date')
        period_no = request.form.get('period_no')
        student_ids = request.form.getlist('student_ids')

        if not date_str or not period_no or not student_ids:
            flash('Please fill in all fields and select at least one student.', 'danger')
            return redirect(url_for('teacher.bulk_attendance'))

        selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        course = Timetable.objects(teacher=teacher_profile, period_no=int(period_no)).first()
        count = 0
        for sid in student_ids:
            student = Student.objects(id=sid).first()
            if student:
                exists = Attendance.objects(student=student, date=selected_date, course=course).first() if course else None
                if not exists:
                    Attendance(
                        student=student,
                        teacher=teacher_profile,
                        course=course,
                        date=selected_date,
                        time=datetime.now(),
                        status='Present',
                        method='Manual'
                    ).save()
                    count += 1

        flash(f'Attendance recorded for {count} student(s).', 'success')
        return redirect(url_for('teacher.dashboard'))

    return render_template('teacher/bulk_attendance.html', students=students)

@teacher_bp.route('/attendance_request', methods=['GET', 'POST'])
def attendance_request():
    """Special permission / attendance request form"""
    teacher_profile = current_user.teacher_profile
    form = AttendanceRequestForm()

    # Populate student choices
    students = Student.objects.all().order_by('name')
    form.student_enrollment.choices = [(str(s.id), f"{s.roll_no} — {s.name}") for s in students]

    # Populate period choices from timetable
    slots = Timetable.objects(teacher=teacher_profile)
    form.period_no.choices = [(str(s.period_no), f"Period {s.period_no} — {s.subject_name}") for s in slots]

    if form.validate_on_submit():
        student = Student.objects(id=form.student_enrollment.data).first()
        if student:
            AttendanceRequest(
                teacher=teacher_profile,
                student=student,
                date=form.date.data,
                period_no=int(form.period_no.data),
                reason=form.reason.data,
                status='Pending'
            ).save()
            flash('Attendance request submitted successfully.', 'success')
            return redirect(url_for('teacher.dashboard'))
        else:
            flash('Student not found.', 'danger')

    return render_template('teacher/request.html', form=form)
