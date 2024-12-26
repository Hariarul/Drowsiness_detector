import streamlit as st
import cv2
from ultralytics import YOLO
import cvzone
import math
import tempfile
import os
from collections import Counter
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Drowsiness & Age Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Drowsiness & Age Detection App"}
)

# Title
st.title("🚗 Drowsiness & Age Detection System")
st.markdown("""
This application detects **drowsiness** of individuals in videos or images based on eye closure.
Upload a file to get started! 💡
""")

# Sidebar
st.sidebar.header("Upload Input")
file = st.sidebar.file_uploader("Upload an Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=False)

# Pre-trained model paths
detection_model_path = "Drowsiness_detector.pt" 
age_model_path = "age_detector.pt"  

# Load models
detection_model = YOLO(detection_model_path)
#st.sidebar.text("Detection model loaded!")
#st.sidebar.text("Loading age model...")
age_model = YOLO(age_model_path)
#st.sidebar.text("Age model loaded!")

# Class names and age labels
classNames = ['awake', 'sleeping', 'Cigarette', 'Phone', 'car']
age_labels = ['0-5', '6-10', '11-15', '16-20', '21-30', '31-40', '41-50', '51-60', '60+']

# Helper function to process frames
def process_frame(frame, detection_model, age_model, frame_count):
    results = detection_model(frame, stream=True)
    detected_boxes = []
    person_states = []
    sleeping_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cls = int(box.cls)
            conf = math.ceil(box.conf * 100) / 100
            currentClass = classNames[cls]

            if conf > 0.3 and currentClass in ['car', 'sleeping']:
                person_states.append(currentClass)

                if currentClass == 'sleeping':
                    sleeping_count += 1

                # Draw bounding box and label
                cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=2, rt=1, colorR=(255, 255, 255), colorC=(255, 255, 255))
                cvzone.putTextRect(frame, f'{currentClass} {conf:.2f}', (x1, y1 - 10), colorR=(255, 0, 255),
                                   colorT=(255, 255, 255), scale=2, thickness=2)

    # Calculate mode of states
    mode_state = Counter(person_states).most_common(1)
    mode_state = mode_state[0][0] if mode_state else "Unknown"

    # Display mode state and sleeping count on frame
    cvzone.putTextRect(frame, f'Mode State: {mode_state}', (50, frame.shape[0] - 60),
                       colorR=(0, 255, 0), colorT=(255, 255, 255), scale=2, thickness=2)
    cvzone.putTextRect(frame, f'People Sleeping: {math.ceil(sleeping_count/2)}', (50, frame.shape[0] - 20),
                       colorR=(255, 0, 0), colorT=(255, 255, 255), scale=2, thickness=2)

    return frame, mode_state, math.ceil(sleeping_count)

# Main app logic
if file is not None:
    file_extension = file.name.split(".")[-1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    tfile.write(file.read())
    tfile.close()

    if file_extension in ["jpg", "jpeg", "png"]:
        img = cv2.imread(tfile.name)
        processed_img, mode_state, sleeping_count = process_frame(img, detection_model, age_model, frame_count=1)
        st.image(processed_img, caption="Processed Image", use_column_width=True)
        st.success(f"Mode State: {mode_state}")
        st.success(f"People Sleeping: {sleeping_count}")

    elif file_extension in ["mp4", "avi", "mov"]:
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            processed_frame, mode_state, sleeping_count = process_frame(frame, detection_model, age_model, frame_count)
            stframe.image(processed_frame, channels="BGR", use_column_width=True)

        cap.release()

    os.remove(tfile.name)

else:
    st.info("👈 Upload a file to get started!")
