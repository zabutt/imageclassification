import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load pre-trained SSD model
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model = hub.load(model_url).signatures['default']

# Streamlit app
st.title("Object Detection Explorer")

# Upload image through Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Function for object detection
def detect_objects(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = tf.image.resize(img_rgb, (300, 300))
    img_array = tf.expand_dims(img_resized, 0)

    detections = model(img_array)

    return detections

# Display object detection
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")

    detections = detect_objects(uploaded_file)

    st.subheader("Object Detection Results:")
    for i in range(detections['detection_boxes'].shape[1]):
        label = int(detections['detection_classes'][0, i].numpy())
        score = detections['detection_scores'][0, i].numpy()
        box = detections['detection_boxes'][0, i].numpy()

        if score > 0.5:  # Consider only high-confidence detections
            st.write(f"Class: {label}, Score: {score:.2f}")

            # Draw bounding box on the image
            img_with_boxes = img_rgb.copy()
            h, w, _ = img_with_boxes.shape
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # Display image with bounding boxes
            st.image(img_with_boxes, caption="Object Detection", use_column_width=True)

