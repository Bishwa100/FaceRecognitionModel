import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
import matplotlib.pyplot as plt


def predict_image_class(model_path, image_path, class_names):
    # Load the saved model
    model = load_model(model_path)

    # Load the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  

    # Make predictions
    predictions = model.predict(img_array)

    # Display the image
    plt.imshow(img_array.squeeze())
    plt.axis('off')
    plt.show()

    # Get the predicted class name
    predicted_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_index]
    print("Predicted class:", predicted_class_name)
    return img_array,predicted_class_name


def check_image_similarity(img_path1, img_array, class_names):
    # Load InceptionV3 model without top layer
    pre_trained_model = InceptionV3(weights='imagenet', include_top=False)

    # Define input layer
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

    # Create the model
    model1 = Model(inputs=[input_1, input_2], outputs=output)

    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Preprocess images
    img1 = image.load_img(img_path1, target_size=(150, 150))
    img_array1 = image.img_to_array(img1)
    img_array1 = np.expand_dims(img_array1, axis=0)
    img_array1 = preprocess_input(img_array1)

    # Predict similarity
    similarity = model1.predict([img_array1, img_array])[0][0]

    # Define a threshold for similarity
    threshold = 0.7

    # Compare similarity with threshold
    if similarity >= threshold:
        return "Images are similar."
    else:
        return "Images are different."


def capture_photo(save_path):
    # Turn on the camera
    camera = cv2.VideoCapture(0)  # 0 is the default camera index, you may need to change it if you have multiple cameras

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture image.")
        return

    # Save the captured frame to a file
    cv2.imwrite(save_path, frame)

    camera.release()
    print("Photo saved successfully.")


save_path = "captured_photo.jpg"
capture_photo(save_path)


model_path = "bishwanathjanaModel.h5"
image_path = "./captured_photo.jpg"
class_names = ["bishwanath", "gobinda", "shivam","shouvik"]
img_array,predicted_class_name = predict_image_class(model_path, image_path, class_names)


img_path1 = f'./{predicted_class_name}/image1.jpg'
result = check_image_similarity(img_path1, img_array, class_names)
print(result)
