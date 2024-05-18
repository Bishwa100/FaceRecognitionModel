import cv2
import numpy as np
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def extract_features(img_array, embedding_model):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = embedding_model.predict(img_array)
    return features.flatten()

def compare_embeddings(anchor_embedding, test_embedding, threshold=0.7):
    distance = np.linalg.norm(anchor_embedding - test_embedding)
    return distance < threshold
