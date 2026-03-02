from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField, DateField, TextAreaField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from models import User

class LoginForm(FlaskForm):
    username = StringField('Identity ID', validators=[DataRequired(), Length(min=2, max=50)])
    password = PasswordField('Security Cipher', validators=[DataRequired()])
    submit = SubmitField('Authenticate')

class RegistrationForm(FlaskForm):
    username = StringField('Login Identity', validators=[DataRequired(), Length(min=2, max=50)])
    full_name = StringField('Full Legal Name', validators=[DataRequired(), Length(min=2, max=150)])
    password = PasswordField('Security Cipher', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Cipher', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Institutional Role', choices=[('student', 'Student'), ('teacher', 'Teacher')], validators=[DataRequired()])
    roll_no = StringField('Government Roll Number', validators=[Length(max=50)])
    department = StringField('Faculty Department', validators=[Length(max=100)])
    submit = SubmitField('Initiate Enrollment')

    def validate_username(self, username):
        user = User.objects(username=username.data).first()
        if user:
            raise ValidationError('This identity ID is already registered.')

class AttendanceRequestForm(FlaskForm):
    student_enrollment = SelectField('Select Delegate', validators=[DataRequired()])
    date = DateField('Session Date', format='%Y-%m-%d', validators=[DataRequired()])
    period_no = SelectField('Time Slot', choices=[], validators=[DataRequired()])
    reason = TextAreaField('Authorization Reason', validators=[DataRequired(), Length(max=200)])
    submit = SubmitField('Submit Authorization')

    def validate_date(self, date):
        from datetime import date as dt_date
        if date.data > dt_date.today():
            raise ValidationError("Future session requests are prohibited.")
