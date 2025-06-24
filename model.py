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

# Load YOLO Model
MODEL_PATH = "/home/sag_umt/yolo_project/yolo_env/runs/detect/train/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# Load TTS Model
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# Define Object Classes for YOLO
CLASS_NAMES = ["stairs", "zebra crossing", "puddles", "vehicles"]

# Streamlit UI
st.title("🔍 YOLO Object Detection & OCR with Speech")
st.sidebar.header("Choose Operation")

# Choose Mode **BEFORE** Uploading Image
mode = st.sidebar.radio("Select Mode:", ("🚗 Object Detection (YOLO)", "🔠 Extract Text (OCR)"))

# Upload Image **ONLY AFTER Mode is Selected**
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

def preprocess_image(image):
    """Preprocess image for YOLO model."""
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def detect_objects(image):
    """Run YOLO model and return detections."""
    results = yolo_model.predict(source=image)
    detections = results[0].boxes  # Extract detected bounding boxes
    object_count = {}

    for detection in detections:
        class_id = int(detection.cls[0])  # Class index
        class_name = CLASS_NAMES[class_id]  # Get class name
        object_count[class_name] = object_count.get(class_name, 0) + 1

    return object_count

def generate_speech(text, output_file="detection_output.wav"):
    """Convert text to speech and save as a WAV file."""
    if not text:
        st.warning("No text detected to convert into speech.")
        return None
    
    audio_data = tts_model.tts(text)
    sf.write(output_file, audio_data, samplerate=22050)
    return output_file

def extract_text(image):
    """Extract text using Tesseract OCR."""
    text = pytesseract.image_to_string(image)
    text = " ".join(text.split())  # Clean up spaces and newlines
    return text

# Process Image **Only if Uploaded**
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if mode == "🚗 Object Detection (YOLO)":
        st.subheader("📸 Object Detection Results")
        preprocessed_image = preprocess_image(image)
        detected_objects = detect_objects(preprocessed_image)

        if detected_objects:
            detected_text = " and ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in detected_objects.items()])
            speech_text = f"{detected_text} ahead."
            st.success(f"🗣 {speech_text}")

            # Generate Speech
            speech_file = generate_speech(speech_text)
            if speech_file:
                st.audio(speech_file, format="audio/wav")
                st.download_button("⬇ Download Detection Audio", data=open(speech_file, "rb"), file_name="detection_output.wav")
        else:
            st.warning("No objects detected.")

    elif mode == "🔠 Extract Text (OCR)":
        st.subheader("📝 Extracted Text")
        extracted_text = extract_text(image)
        st.text_area("Detected Text", extracted_text, height=150)

        if extracted_text:
            # Generate Speech
            speech_file = generate_speech(extracted_text, "ocr_output.wav")
            if speech_file:
                st.audio(speech_file, format="audio/wav")
                st.download_button("⬇ Download OCR Audio", data=open(speech_file, "rb"), file_name="ocr_output.wav")
        else:
            st.warning("No text detected in the image.")
