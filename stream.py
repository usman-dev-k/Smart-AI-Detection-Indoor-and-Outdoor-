import os
import cv2
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pytesseract
import soundfile as sf
from TTS.api import TTS

# === Configs ===
INDOOR_MODEL_PATH = "/home/sag_umt/yolo_project/yolov8s.pt"
OUTDOOR_MODEL_PATH = "/home/sag_umt/Music/outdoor_best_weights/weights/best.pt"
# === Outdoor model class labels (10 classes) ===
OUTDOOR_CLASSES = [
    'Ambulance', 'Auto-Rikshaw', 'bike', 'bus', 'car',
    'puddle', 'stairs', 'truck', 'van', 'zebra-crossing'
]

# === Load TTS model ===
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# === Helper Functions ===

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def detect_objects(image, model, class_names):
    results = model.predict(source=image)
    detections = results[0].boxes
    object_count = {}

    for detection in detections:
        class_id = int(detection.cls[0])
        if class_id >= len(class_names):
            continue
        class_name = class_names[class_id]
        object_count[class_name] = object_count.get(class_name, 0) + 1

    return object_count

def generate_speech(text, output_file="output.wav"):
    if not text:
        st.warning("No text to convert into speech.")
        return None
    audio_data = tts_model.tts(text)
    sf.write(output_file, audio_data, samplerate=22050)
    return output_file

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return " ".join(text.split())

# === Streamlit App UI ===

st.title("🧠 Smart Assistive Vision System")

# Tabs
tab1, tab2 = st.tabs(["📷 Object Detection", "🔠 OCR + TTS"])

# --- Tab 1: Object Detection ---
with tab1:
    st.subheader("Choose Environment")
    env = st.radio("Environment Type", ["Indoor", "Outdoor"], horizontal=True)

    uploaded_file = st.file_uploader("Upload an image for object detection...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model and classes
        if env == "Indoor":
            model_path = INDOOR_MODEL_PATH
            yolo_model = YOLO(model_path)
            class_names = yolo_model.names  # Use built-in COCO class names
        else:
            model_path = OUTDOOR_MODEL_PATH
            yolo_model = YOLO(model_path)
            class_names = OUTDOOR_CLASSES  # Custom class list

        preprocessed = preprocess_image(image)
        detections = detect_objects(preprocessed, yolo_model, class_names)

        if detections:
            detected_text = " and ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in detections.items()])
            speech_text = f"{detected_text} ahead."
            st.success(f"🗣 {speech_text}")

            audio_file = generate_speech(speech_text, "detection_output.wav")
            if audio_file:
                st.audio(audio_file, format="audio/wav")
                st.download_button("⬇ Download Detection Audio", data=open(audio_file, "rb"), file_name="detection_output.wav")
        else:
            st.warning("No objects detected.")

# --- Tab 2: OCR + TTS ---
with tab2:
    st.subheader("Upload Image for OCR")
    uploaded_file_ocr = st.file_uploader("Upload an image for OCR...", type=["jpg", "jpeg", "png"], key="ocr")

    if uploaded_file_ocr:
        image = Image.open(uploaded_file_ocr)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        text = extract_text(image)
        st.text_area("Extracted Text", text, height=150)

        if text:
            audio_file = generate_speech(text, "ocr_output.wav")
            if audio_file:
                st.audio(audio_file, format="audio/wav")
                st.download_button("⬇ Download OCR Audio", data=open(audio_file, "rb"), file_name="ocr_output.wav")
        else:
            st.warning("No text detected in the image.")
