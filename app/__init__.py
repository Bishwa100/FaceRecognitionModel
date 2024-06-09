from flask import Flask
from .utils import load_models
from .config import get_db_connection
from .routes import main  # Import the blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.config['DB_CONNECTION'] = get_db_connection()
    
    # Register the blueprint
    app.register_blueprint(main)

    with app.app_context():
        load_models(app)

    return app
