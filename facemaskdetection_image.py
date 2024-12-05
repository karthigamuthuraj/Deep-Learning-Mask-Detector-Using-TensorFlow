import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

# Function to load models
@st.cache_resource
def load_models(face_model_path, mask_model_path):
    # Load face detector model
    prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face_model_path, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # Load face mask detector model
    model = load_model(mask_model_path)
    return net, model

# Function to detect and display results
def detect_mask(image, net, model, confidence_threshold=0.5):
    orig = image.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return image

# Streamlit app interface
st.title("Face Mask Detection with TensorFlow")
st.write("Upload an image to detect if people are wearing masks or not.")

# Sidebar for model configurations
face_model_dir = st.sidebar.text_input("Path to Face Detector Model", "face_detector")
mask_model_path = st.sidebar.text_input("Path to Mask Detector Model", "mask_detector.model")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Load models
net, model = load_models(face_model_dir, mask_model_path)

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Detect mask
    result_image = detect_mask(image, net, model, confidence_threshold)
    
    # Convert result image to RGB for displaying
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    # Display images
    st.image(result_image, caption="Processed Image", use_column_width=True)
