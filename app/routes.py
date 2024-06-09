from flask import request, jsonify, Blueprint
from .utils import process_frame
from .config import get_db_connection

main = Blueprint('main', __name__)

@main.route('/students', methods=['GET'])
def get_students():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT student_id, first_name, last_name, birth_date, enrollment_date, major FROM student_data.student")
        students = cur.fetchall()
        cur.close()
        conn.close()
        student_list = []
        for student in students:
            student_dict = {
                'student_id': student[0],
                'first_name': student[1],
                'last_name': student[2],
                'birth_date': student[3],
                'enrollment_date': student[4],
                'major': student[5]
            }
            student_list.append(student_dict)
        
        return student_list
    except Exception as e:
        return jsonify({'error': str(e)}), 500
