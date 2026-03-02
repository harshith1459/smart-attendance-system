from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from services.attendance_service import AttendanceService
from models import Student, Timetable, Attendance, AttendanceRequest
from forms import AttendanceRequestForm
from datetime import datetime, date
import base64
import os

try:
    import cv2
    import numpy as np
    from ml.recognize import FaceRecognizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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

        if not ML_AVAILABLE:
            return jsonify({'status': 'error', 'message': 'ML libraries (cv2/numpy) not installed.'})

        # Decode the base64 image from the camera
        try:
            header, encoded = image_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Image decode failed: {str(e)}'})

        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data.'})

        # Use real face recognition
        db_path = os.path.join(os.getcwd(), 'static', 'uploads', 'dataset')
        recognizer = FaceRecognizer(db_path)
        results = recognizer.recognize_face(img)

        teacher_profile = current_user.teacher_profile
        current_course = AttendanceService.get_current_slot(teacher_profile)

        recognized_rolls = []
        total_detected = len(results)

        for (student_id, score, box) in results:
            if student_id == 'unknown':
                continue
            student = Student.objects(id=student_id).first()
            if student and student.roll_no not in recognized_rolls:
                recognized_rolls.append(student.roll_no)

        return jsonify({
            'status': 'success',
            'total_detected': total_detected,
            'names': recognized_rolls,
            'subject': current_course.subject_name if current_course else 'General Session'
        })

    return render_template('teacher/attendance.html', ml_available=ML_AVAILABLE)


@teacher_bp.route('/save_attendance', methods=['POST'])
def save_attendance():
    """Save attendance for the recognized students (called after scan)"""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received.'})

    roll_numbers = data.get('roll_numbers', [])
    if not roll_numbers:
        return jsonify({'status': 'error', 'message': 'No students to mark.'})

    teacher_profile = current_user.teacher_profile
    current_course = AttendanceService.get_current_slot(teacher_profile)

    saved_count = 0
    for roll in roll_numbers:
        student = Student.objects(roll_no=roll).first()
        if student:
            if AttendanceService.record_attendance(student, teacher_profile, current_course):
                saved_count += 1

    return jsonify({
        'status': 'success',
        'message': f'Attendance marked for {saved_count} student(s).',
        'saved_count': saved_count
    })

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
