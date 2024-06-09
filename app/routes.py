from flask import request, jsonify, Blueprint
from .utils import process_frame
from .config import get_db_connection
from .model import *
from .dal import *
main = Blueprint('main', __name__)

@main.route('/students', methods=['GET'])
def get_students():
    try:
        success, students = get_all_students()
        if success:
            student_list = [create_student_dict(student) for student in students]
            return student_list,200
        else:
            return jsonify({'error': 'Failed to get students'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@main.route('/add/students', methods=['POST'])
def add_student():
    try:
        student_data = request.json
        
        required_fields = ['student_code', 'first_name', 'last_name', 'birth_date', 'enrollment_date', 'major']
        for field in required_fields:
            if field not in student_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        success = insert_student(student_data)
        
        if success:
            return jsonify({'message': 'Student added successfully'}), 201
        else:
            return jsonify({'error': 'Failed to add student'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@main.route('/attendance', methods=['GET'])
def attendance():
    student_code = request.args.get('student_code')
    attendance_date = request.args.get('attendance_date')

    if not student_code or not attendance_date:
        return jsonify({'error': 'Please provide both student_code and attendance_date'}), 400

    success, data = get_student_attendance(student_code, attendance_date)

    if not success:
        return jsonify({'error': data}), 500
    else:
        student_list = [create_student_dict(student) for student in data]
        return student_list,200


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
