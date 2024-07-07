import os
from flask import Flask, render_template
from .utils import load_models
from .config import get_db_connection
from .routes import main

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    app.config['DB_CONNECTION'] = get_db_connection()
    
    # Specify the path to the templates folder
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    app.template_folder = template_path
    
    app.register_blueprint(main)

    with app.app_context():
        load_models(app)

    return app
