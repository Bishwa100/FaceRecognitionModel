import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class Config:
    DEBUG = True

def get_db_connection():
    conn = psycopg2.connect(
        database=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST'),  
        port=os.getenv('POSTGRES_PORT')  
    )
    return conn
