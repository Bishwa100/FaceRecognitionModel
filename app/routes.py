from flask import request, jsonify, Blueprint,send_file,Response,render_template,current_app
from .utils import *
from .config import get_db_connection
from .model import *
from .dal import *
import flask_excel as excel
import pandas as pd
import io
import cv2
import numpy as np


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

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
    isExcel = request.args.get('isExcel')

    success, data = get_student_attendance(student_code, attendance_date)

    if not success:
        return jsonify({'error': data}), 500

    student_list = [create_student_attendance_dict(student) for student in data]

    if isExcel:
        try:
            df = pd.DataFrame(student_list)  # Create DataFrame from student_list
            output = io.BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(output, mimetype='text/csv', as_attachment=True, download_name='attendance.csv'), 200
        except Exception as e:
            return jsonify({'error': 'Failed to generate Excel file', 'details': str(e)}), 500
    else:
        return jsonify(student_list), 200
    

    # @main.route('/attendance', methods=['POST'])
# def mark_attendance():
#     data = request.json
#     student_id = data['student_id']
#     attendance = Attendance(student_id=student_id)
#     db.session.add(attendance)
#     db.session.commit()
#     return jsonify({'message': 'Attendance marked'})


@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/predict_camera', methods=['GET'])
def predict_camera():
    model_path = "model.h5"  # Path to your face recognition model
    embedding_model = current_app.config['embedding_model']
    class_names = current_app.config['CLASS_NAMES']

    # Check if the model file exists
    if model_path not in os.listdir('.'):
        return jsonify({"message": "Model file not found. Please ensure the model.h5 file is in the correct directory."}), 404

    predictions = capture_and_predict(model_path, class_names, embedding_model)

    if "error" in predictions:
        return jsonify({"message": predictions["error"]}), 500

    return jsonify({"message": "Camera prediction completed", "predictions": predictions})

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()


