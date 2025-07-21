import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pytesseract
from gtts import gTTS
from playsound import playsound
import tempfile
from ultralytics import YOLO
import time

# Load models
indoor_model = YOLO("models/indoor.pt")
outdoor_model = YOLO("models/outdoor.pt")

# Helper functions
def speak_text(text):
    if not text.strip():
        return
    tts = gTTS(text=text)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        playsound(fp.name)

def detect_objects(frame, model):
    results = model(frame)
    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        names = r.names
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(classes[i])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            speak_text(label)
    return frame

def run_ocr(image):
    text = pytesseract.image_to_string(image)
    return text

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Smart Camera App")

tab1, tab2 = st.tabs(["📷 Real-Time Object Detection", "📖 OCR - Text to Speech"])

# --- TAB 1: Object Detection ---
with tab1:
    st.header("Real-Time Object Detection")
    detection_mode = st.radio("Select Mode", ["Indoor", "Outdoor"])

    start_camera = st.button("Start Detection")

    if start_camera:
        model = indoor_model if detection_mode == "Indoor" else outdoor_model
        cap = cv2.VideoCapture(0)

        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the webcam.")
                break

            frame = cv2.resize(frame, (640, 480))
            frame = detect_objects(frame, model)
            st_frame.image(frame, channels="BGR")

            if st.button("Stop"):
                cap.release()
                break

# --- TAB 2: OCR ---
with tab2:
    st.header("OCR - Click Image and Convert to Speech")
    cap = cv2.VideoCapture(0)
    st_camera = st.empty()
    capture_btn = st.button("Click Picture")

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            st_camera.image(frame, channels="BGR")

    if capture_btn:
        if not ret:
            st.error("Camera frame not available.")
        else:
            image = frame
            st.image(image, caption="Captured Image", channels="BGR")

            text = run_ocr(image)
            st.text_area("Detected Text", text, height=200)

            st.success("Speaking Text...")
            speak_text(text)

    cap.release()
