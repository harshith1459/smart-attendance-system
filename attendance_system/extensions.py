import mongoengine as db
from flask_login import LoginManager

# Core Extensions
login_manager = LoginManager()

# Configuration
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'
