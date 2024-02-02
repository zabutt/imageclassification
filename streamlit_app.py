import streamlit as st
import cv2
import numpy as np

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Streamlit app
st.title("Vehicle Counting App")

# Upload video through Streamlit file uploader
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

# Function for vehicle counting
def count_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Prepare frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        output_layer_names = net.getUnconnectedOutLayersNames()

        # Run YOLO forward pass
        detections = net.forward(output_layer_names)

        vehicle_count = 0

        for obj in detections[0]:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to vehicles in YOLO
                vehicle_count += 1

        # Draw vehicle count on the frame
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        st.image(frame, caption="Vehicle Counting", use_column_width=True)

    cap.release()

# Display vehicle counting
if uploaded_file is not None:
    st.video(uploaded_file)

    count_vehicles(uploaded_file)
