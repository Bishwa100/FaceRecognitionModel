import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import backend as K
import gdown

# Define the triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    total_length = y_pred.shape.as_list()[-1]
    anchor = y_pred[:, 0:int(total_length/3)]
    positive = y_pred[:, int(total_length/3):int(2*total_length/3)]
    negative = y_pred[:, int(2*total_length/3):total_length]
    
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss

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

# Capture images from the webcam and predict
def capture_and_predict(model_path, class_names, embedding_model):
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
            print(f"Predicted class Name: {predicted_class_name}")

            # Load the reference image for similarity check
            img_path1 = f'./{predicted_class_name}/image1.jpg'
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

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def download_model_from_google_drive(drive_link, output):
    file_id = drive_link.split('/')[-2]
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
    return output

# Google Drive link for the model
# google_drive_link = "https://drive.google.com/file/d/1wHQHMeAPQK-_gjIbJRPnt-oXTKdIZgim/view?usp=sharing"
# output_path = "model.h5"
# model_path = download_model_from_google_drive(google_drive_link, output_path)

model_path = "bishwanathjanaModel.h5"
class_names = ["bishwanath", "gobinda", "shivam", "shouvik"]

input_shape = (150, 150, 3)
embedding_model = create_embedding_model(input_shape)

capture_and_predict(model_path, class_names, embedding_model)
