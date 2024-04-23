from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3

app = Flask(__name__)

# Load the classification model
classification_model_path = "bishwanathjanaModel.h5"
classification_model = load_model(classification_model_path)

# Load InceptionV3 model without top layer for similarity checking
pre_trained_model = InceptionV3(weights='imagenet', include_top=False)

# Define input shape
input_shape = (150, 150, 3)
input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)

# Extract features from the base model
features_1 = pre_trained_model(input_1)
features_2 = pre_trained_model(input_2)

# Flatten the features
flatten_1 = Flatten()(features_1)
flatten_2 = Flatten()(features_2)

# Concatenate the flattened features
concatenated_features = Concatenate()([flatten_1, flatten_2])

# Add a dense layer for binary classification
dense_layer = Dense(512, activation='relu')(concatenated_features)
output = Dense(1, activation='sigmoid')(dense_layer)

# Create the similarity checking model
similarity_model = Model(inputs=[input_1, input_2], outputs=output)
similarity_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ["bishwanath", "gobinda", "shivam", "shouvik"]

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image
        image_path = "uploaded_image.jpg"
        file.save(image_path)

        # Load the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make predictions
        predictions = classification_model.predict(img_array)

        # Get the predicted class name
        predicted_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_index]

        return jsonify({'predicted_class': predicted_class_name})

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image
        image_path = "uploaded_image.jpg"
        file.save(image_path)

        # Preprocess the images
        img1 = image.load_img(img_path1, target_size=(150, 150))
        img_array1 = image.img_to_array(img1)
        img_array1 = np.expand_dims(img_array1, axis=0)
        img_array1 = preprocess_input(img_array1)

        img2 = image.load_img(image_path, target_size=(150, 150))
        img_array2 = image.img_to_array(img2)
        img_array2 = np.expand_dims(img_array2, axis=0)
        img_array2 = preprocess_input(img_array2)

        # Predict similarity
        similarity = similarity_model.predict([img_array1, img_array2])[0][0]

        # Define a threshold for similarity
        threshold = 0.7

        # Compare similarity with threshold
        if similarity >= threshold:
            return jsonify({'result': 'Images are similar'})
        else:
            return jsonify({'result': 'Images are different'})

if __name__ == '__main__':
    app.run(debug=True)
