from flask import request, jsonify, current_app, Blueprint
from . import db
from .model import Student, Attendance
from .utils import process_frame

main = Blueprint('main', __name__)

@main.route('/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    return jsonify([{'id': student.id, 'name': student.name} for student in students])

@main.route('/attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    student_id = data['student_id']
    attendance = Attendance(student_id=student_id)
    db.session.add(attendance)
    db.session.commit()
    return jsonify({'message': 'Attendance marked'})

@main.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    predictions = process_frame(frame, current_app.config['CLASS_NAMES'])

    return jsonify(predictions)

# Ensure embedding model is loaded before any request
@main.before_app_request
def before_request():
    if not hasattr(current_app, 'embedding_model'):
        current_app.embedding_model = load_embedding_model()
