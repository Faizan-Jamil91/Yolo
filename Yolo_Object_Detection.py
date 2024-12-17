import torch
import pyttsx3
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speed of speech

# Load the pre-trained YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to perform object detection and TTS
def detect_objects(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(image_rgb)

    # Extract the results
    labels, coordinates = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    detected_objects = set()  # Store detected object names to avoid repetitive announcements
    height, width, _ = frame.shape

    # Draw bounding boxes and labels on the frame
    for i in range(len(labels)):
        row = coordinates[i]
        x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
        label = model.names[int(labels[i])]
        detected_objects.add(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Announce detected objects
    if detected_objects:
        announcement = "Detected: " + ", ".join(detected_objects)
        engine.say(announcement)
        engine.runAndWait()

    return frame, detected_objects

# Streamlit app layout
st.set_page_config(page_title="YOLOv5 Real-Time Object Detection", page_icon=":guardsman:", layout="centered")
st.title("YOLOv5 Real-Time Object Detection App")
st.markdown("""
    **Welcome to the YOLOv5 Real-Time Object Detection App!**  
    This application uses the YOLOv5 deep learning model to detect objects from a real-time webcam feed.
    The detected objects will be highlighted with bounding boxes and labels, and an audio announcement will be made.
""")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/94/Ultralytics_logo_black_on_white.svg", use_container_width=True)

st.sidebar.markdown("### How to Use")
st.sidebar.markdown("""
1. Click 'Start' to begin real-time detection using your webcam.
2. Click 'Stop' to end the detection and release the webcam.
3. The app will process the video feed and detect objects in real-time.
4. Detected objects will be highlighted and announced audibly.
""")

# Start and Stop buttons
start_button = st.button("Start")
stop_button = st.button("Stop")

if start_button:
    # Start the video capture
    stframe = st.empty()  # Empty frame to display the webcam feed

    # Use OpenCV to capture from the webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while True:
            ret, frame = cap.read()

            if not ret:
                st.error("Failed to grab frame")
                break

            # Perform object detection on the captured frame
            detected_frame, detected_objects = detect_objects(frame)

            # Display the resulting frame in the Streamlit app
            stframe.image(detected_frame, channels="BGR", use_container_width=True)

            # Display detected objects
            if detected_objects:
                st.sidebar.subheader("Detected Objects:")
                st.sidebar.write(", ".join(detected_objects))

            time.sleep(0.1)  # Adding slight delay to ensure smooth video stream

        cap.release()
        cv2.destroyAllWindows()

elif stop_button:
    st.empty()  # Remove the frame when stop is clicked
    st.success("Detection Stopped.")

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit and YOLOv5.  
[GitHub Repo](https://github.com/ultralytics/yolov5)
""")

