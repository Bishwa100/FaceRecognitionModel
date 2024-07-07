from flask import Flask, jsonify, current_app
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import os
import gdown

app = Flask(__name__)

# Define the embedding model
def create_embedding_model(input_shape):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    flatten = Flatten()(base_model.output)
    embedding_model = Model(base_model.input, flatten)
    return embedding_model

# Detect faces in the frame
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Predict the class of the image
def predict_image_class(model, img_array, class_names):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_index]
    return predicted_class_name

# Extract features using the embedding model
def extract_features(img_array, embedding_model):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = embedding_model.predict(img_array)
    return features.flatten()

# Compare embeddings using a threshold
def compare_embeddings(anchor_embedding, test_embedding, threshold=0.7):
    distance = np.linalg.norm(anchor_embedding - test_embedding)
    return distance < threshold

def download_model_from_google_drive(drive_link, output):
    file_id = drive_link.split('/')[-2]
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    return output

def load_models(app):
    google_drive_link = "https://drive.google.com/file/d/1wHQHMeAPQK-_gjIbJRPnt-oXTKdIZgim/view?usp=sharing"
    model_filename = "model.h5"
    if not os.path.exists(model_filename):
        model_path = download_model_from_google_drive(google_drive_link, model_filename)
    else:
        model_path = model_filename
    face_recognition_model = load_model(model_path)
    input_shape = (150, 150, 3)
    embedding_model = create_embedding_model(input_shape)
    app.config['face_recognition_model'] = face_recognition_model
    app.config['embedding_model'] = embedding_model
    app.config['CLASS_NAMES'] = ["bishwanath", "gobinda", "shivam", "shouvik"]

def capture_and_predict(model_path, class_names, embedding_model):
    model = load_model(model_path)
    camera = cv2.VideoCapture(0)  # Open the default camera

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return jsonify({"error": "Could not open camera"}), 500

    predictions = []

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        faces = detect_face(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (150, 150))
            img_array = image.img_to_array(face_img)
            img_array /= 255.0

            predicted_class_name = predict_image_class(model, img_array, class_names)
            print(f"Predicted class Name: {predicted_class_name}")

            # Load the reference image for similarity check
            img_path1 = f'./{predicted_class_name}/image1.jpg'  # Update this path as per your directory structure
            img1 = image.load_img(img_path1, target_size=(150, 150))
            img_array1 = image.img_to_array(img1)

            anchor_embedding = extract_features(img_array1, embedding_model)
            test_embedding = extract_features(img_array, embedding_model)
            
            result = compare_embeddings(anchor_embedding, test_embedding)
            if result:
                label = predicted_class_name
            else:
                label = "Unknown"

            print(f"Similarity Check Result: {result} - Label: {label}")

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            predictions.append({"bbox": (x, y, w, h), "label": label})

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

    return predictions