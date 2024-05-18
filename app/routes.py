from flask import request, jsonify, current_app
from . import db
from .models import Student, Attendance
from .utils import detect_face, extract_features, compare_embeddings
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import base64
import numpy as np
import os

embedding_model = None
input_shape = (150, 150, 3)

def load_embedding_model():
    global embedding_model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    flatten = Flatten()(base_model.output)
    embedding_model = Model(base_model.input, flatten)

@app.route('/students', methods=['GET'])
def get_students():
    students = Student.query.all()
    return jsonify([{'id': student.id, 'name': student.name} for student in students])

@app.route('/attendance', methods=['POST'])
def mark_attendance():
    data = request.json
    student_id = data['student_id']
    attendance = Attendance(student_id=student_id)
    db.session.add(attendance)
    db.session.commit()
    return jsonify({'message': 'Attendance marked'})

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    data = request.json
    image_data = data['image']
    decoded_image = base64.b64decode(image_data)
    np_image = np.frombuffer(decoded_image, dtype=np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    faces = detect_face(frame)
    face_images = []

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150, 150))
        img_array = image.img_to_array(face_img)
        img_array /= 255.0

        predicted_class_name = predict_image_class(model, img_array, class_names)

        img_path1 = f'./{predicted_class_name}/image1.jpg'
        img1 = image.load_img(img_path1, target_size=(150, 150))
        img_array1 = image.img_to_array(img1)

        anchor_embedding = extract_features(img_array1, embedding_model)
        test_embedding = extract_features(img_array, embedding_model)

        result = compare_embeddings(anchor_embedding, test_embedding)
        label = predicted_class_name if result else "Unknown"

        face_data = {
            "face": base64.b64encode(cv2.imencode('.jpg', face_img)[1]).decode('utf-8'),
            "label": label
        }
        face_images.append(face_data)

    return jsonify({'faces': face_images})

if embedding_model is None:
    load_embedding_model()
