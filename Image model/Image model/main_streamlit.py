import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/unzipped/converted_keras/keras_model.h5", compile=True)

# Load the labels
class_names = open("C:/unzipped/converted_keras/labels.txt", "r").readlines()

# Open the webcam
camera = cv2.VideoCapture(0)

# Streamlit app header
st.title("Webcam Image Classification")

# Run the Streamlit app
while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display prediction and confidence score in the Streamlit app
    st.write("Class:", class_name[2:])
    st.write("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Display the webcam image in the Streamlit app
    # Normalize the image array
    image = (image - image.min()) / (image.max() - image.min())

    # Display the webcam image in the Streamlit app
    st.image(image, channels="BGR", use_column_width=True)

    

camera.release()
st.write("Webcam Stopped")

