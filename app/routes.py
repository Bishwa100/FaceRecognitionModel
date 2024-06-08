from flask import request, jsonify, current_app, Blueprint
from .utils import process_frame
from .config import get_db_connection

main = Blueprint('main', __name__)

@main.route('/students', methods=['GET'])
def get_students():
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


# @main.route('/attendance', methods=['POST'])
# def mark_attendance():
#     data = request.json
#     student_id = data['student_id']
#     attendance = Attendance(student_id=student_id)
#     db.session.add(attendance)
#     db.session.commit()
#     return jsonify({'message': 'Attendance marked'})

# @main.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     file_bytes = np.frombuffer(file.read(), np.uint8)
#     frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     predictions = process_frame(frame, current_app.config['CLASS_NAMES'])

#     return jsonify(predictions)

# # Ensure embedding model is loaded before any request
# @main.before_app_first_request
# def before_first_request():
#     current_app.config['embedding_model'] = load_embedding_model()

