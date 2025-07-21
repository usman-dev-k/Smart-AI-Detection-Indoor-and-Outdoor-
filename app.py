import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pytesseract
from gtts import gTTS
import tempfile
import os

# Title
st.set_page_config(page_title="YOLO Object Detection + OCR + Speech")
st.title("YOLOv8 Object Detection with OCR and Text-to-Speech")

# Load YOLO models
indoor_model = YOLO("models/indoor.pt")
outdoor_model = YOLO("models/outdoor.pt")

# Model selection
model_type = st.selectbox("Choose model type:", ["Indoor", "Outdoor"])
model = indoor_model if model_type == "Indoor" else outdoor_model

# Mode selection
mode = st.radio("Choose input mode:", ["Upload Image", "Use Webcam"])

# Function: Object Detection
def detect_objects(image, model):
    results = model(image)
    res_plotted = results[0].plot()
    return res_plotted, results[0].boxes.cls.cpu().numpy()

# Function: OCR
def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Function: Text-to-Speech
def speak_text(text):
    if not text.strip():
        st.warning("No text to speak.")
        return
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# Upload Mode
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Object Detection"):
            result_img, _ = detect_objects(image_np, model)
            st.image(result_img, caption="Detection Result", use_column_width=True)

        if st.button("Run OCR"):
            text = perform_ocr(image_np)
            st.text_area("Detected Text:", text, height=150)
            speak_text(text)

# Webcam Mode
elif mode == "Use Webcam":
    st.warning("Webcam mode: Click 'Capture' to take a frame")
    if st.button("Capture from Webcam"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.image(frame, channels="BGR", caption="Captured Frame")

            result_img, _ = detect_objects(frame, model)
            st.image(result_img, caption="Detection Result")

            if st.button("Run OCR on Captured Frame"):
                text = perform_ocr(frame)
                st.text_area("Detected Text:", text, height=150)
                speak_text(text)
        else:
            st.error("Failed to capture frame from webcam.")
