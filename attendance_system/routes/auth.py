from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from models import User
from forms import LoginForm, RegistrationForm
from extensions import login_manager
from services.auth_service import AuthService

auth_bp = Blueprint('auth', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.objects(pk=user_id).first()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
            
    form = LoginForm()
    if form.validate_on_submit():
        user, error = AuthService.authenticate(form.username.data, form.password.data)
        if user:
            login_user(user)
            flash(f'Identity verification successful. Welcome, {user.username}.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.dashboard'))
        else:
            flash(error, 'danger')
            
    return render_template('auth/login.html', title='Authentication Gate', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
        
    form = RegistrationForm()
    if form.validate_on_submit():
        AuthService.register_user(form.data)
        flash('Enrollment request submitted. Please wait for administrative approval.', 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html', title='Identity Enrollment', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Identity session terminated.', 'info')
    return redirect(url_for('auth.login'))
