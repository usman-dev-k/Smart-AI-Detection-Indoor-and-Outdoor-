import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import pytesseract
from PIL import Image
import tempfile
import time

# Streamlit UI setup
st.set_page_config(page_title="Real-Time YOLO + OCR")
st.title("🎯 Real-Time Object Detection + OCR")

# Load models
indoor_model = YOLO("models/indoor.pt")
outdoor_model = YOLO("models/outdoor.pt")

# Select model
model_type = st.selectbox("Choose YOLO model:", ["Indoor", "Outdoor"])
model = indoor_model if model_type == "Indoor" else outdoor_model

# OCR flag
enable_ocr = st.checkbox("Enable OCR separately", value=False)

# Start camera
start = st.button("Start Real-Time Detection")

frame_placeholder = st.empty()

def detect_objects(frame, model):
    results = model(frame)
    return results[0].plot()

def run_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def speak_text(text):
    if text.strip() == "":
        st.warning("No text detected.")
        return
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# Camera loop
if start:
    cap = cv2.VideoCapture(0)
    st.info("Press Stop or use Ctrl+C to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        output_frame = detect_objects(frame, model)

        frame_placeholder.image(output_frame, channels="BGR", use_column_width=True)

        # OCR
        if enable_ocr:
            text = run_ocr(frame)
            st.text_area("📝 OCR Output", text, height=150)
            speak_text(text)
            time.sleep(3)  # pause to avoid TTS spamming

        # Optional delay (streamlit needs time to render)
        time.sleep(0.03)

    cap.release()
    st.success("Camera stopped.")
