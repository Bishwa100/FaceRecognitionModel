from flask import Flask
from .utils import load_models
from .config import get_db_connection

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.config['DB_CONNECTION'] = get_db_connection()
    
    with app.app_context():
        from . import routes

        load_models(app)

    return app
