import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from scipy.spatial.distance import cosine

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def predict_image_class(model, img_array, class_names):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_index]
    return predicted_class_name

def extract_features(img_array):
    pre_trained_model = InceptionV3(weights='imagenet', include_top=False)
    input_shape = (150, 150, 3)
    input_ = Input(shape=input_shape)
    features = pre_trained_model(input_)
    flatten = Flatten()(features)
    feature_model = Model(inputs=input_, outputs=flatten)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_model.predict(img_array)
    return features.flatten() 


def compare_images(img_array1, img_array2, threshold=0.7):
    features1 = extract_features(img_array1)
    features2 = extract_features(img_array2)
    similarity = 1 - cosine(features1, features2)
    if similarity >= threshold:
        return True
    else:
        return False

def capture_and_predict(model_path, class_names):
    model = load_model(model_path)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

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

            # Load the reference image for similarity check
            img_path1 = f'./{predicted_class_name}/image1.jpg'
            img1 = image.load_img(img_path1, target_size=(150, 150))
            img_array1 = image.img_to_array(img1)

            result = compare_images(img_array1, img_array)
            if result:
                label = predicted_class_name
            else:
                label = "Unknown"

            print(f"Similarity Check Result: {result}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

model_path = "bishwanathjanaModel.h5"
class_names = ["bishwanath", "gobinda", "shivam", "shouvik"]
capture_and_predict(model_path, class_names)
