from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import current_user
from services.student_service import StudentService

student_bp = Blueprint('student', __name__)

@student_bp.before_request
def student_required():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    if current_user.role != 'student':
        flash('Security Alert: Student credentials required.', 'danger')
        return redirect(url_for('auth.login'))

@student_bp.route('/dashboard')
def dashboard():
    student_profile = current_user.student_profile
    if not student_profile:
        flash('Identity error: Student profile not found.', 'danger')
        return redirect(url_for('auth.logout'))
        
    stats = StudentService.get_performance_stats(student_profile)
    history = StudentService.get_attendance_history(student_profile)
    
    return render_template('student/dashboard.html', 
                           student=student_profile, 
                           stats=stats, 
                           history=history)

@student_bp.route('/enroll_face_live', methods=['POST'])
def enroll_face_live():
    from flask import request, jsonify
    data = request.json
    image_data = data.get('image')
    index = data.get('index', 0)
    
    if image_data:
        student_profile = current_user.student_profile
        StudentService.save_biometric_frame(student_profile, image_data, index)
        return jsonify({
            'status': 'success', 
            'message': f'Identity Packet {index + 1}/5 Synchronized' if index < 4 else 'Enrollment Complete'
        })
    
    return jsonify({'status': 'error', 'message': 'Packet loss: No identity data received.'})
