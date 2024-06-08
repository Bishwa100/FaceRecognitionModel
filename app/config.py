import os
import psycopg2

class Config:
    DEBUG = True

def get_db_connection():
    conn = psycopg2.connect(
        database=os.getenv('POSTGRES_DB', 'mydatabase'),
        user=os.getenv('POSTGRES_USER', 'user'),
        password=os.getenv('POSTGRES_PASSWORD', 'password'),
        host=os.getenv('POSTGRES_HOST', 'db'),  # Default to 'db'
        port=os.getenv('POSTGRES_PORT', '5432')  # Default to '5432'
    )
    return conn
