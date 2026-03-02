from flask import Flask, redirect, url_for, render_template
from config import Config
from extensions import db, login_manager
from routes import auth_bp, admin_bp, teacher_bp, student_bp, main_bp
from mongoengine import connect
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Database Connection
    settings = app.config.get('MONGODB_SETTINGS', {})
    connect(host=settings.get('host', 'mongodb://localhost:27017/attendance_db'))

    # Initialize extensions
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(teacher_bp, url_prefix='/teacher')
    app.register_blueprint(student_bp, url_prefix='/student')

    # Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
