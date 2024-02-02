import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load YOLOv4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Title
st.title("YOLOv4-tiny Object Detection")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # YOLOv4-tiny Object Detection
    height, width, _ = img_array.shape
    blob = cv2.dnn.blobFromImage(img_array, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    # Display the results
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w/2), int(center_y - h/2)
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"Class {class_id}, Confidence: {round(confidence * 100, 2)}%"
                cv2.putText(img_array, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    st.image(img_array, channels="BGR", caption="YOLOv4-tiny Object Detection", use_column_width=True)

   
