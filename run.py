from app import create_app
from app.config import Config
from dotenv import load_dotenv, find_dotenv
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
