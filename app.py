import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pytesseract
from gtts import gTTS
from ultralytics import YOLO
import tempfile

# Load YOLOv8 models
indoor_model = YOLO("indoor.pt")
outdoor_model = YOLO("outdoor.pt")

st.set_page_config(layout="wide")
st.title("📷 Smart Camera App (Streamlit Cloud)")

tab1, tab2 = st.tabs(["📦 Object Detection", "🔎 OCR + Text to Speech"])

def detect_objects(image, model):
    frame = np.array(image)
    results = model(frame)
    annotated_frame = results[0].plot()
    labels = results[0].names
    spoken = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = labels[class_id]
        if label not in spoken:
            spoken.append(label)

    return annotated_frame, ", ".join(spoken)

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Tab 1: Object Detection
with tab1:
    st.header("Upload Image for Object Detection")
    mode = st.selectbox("Select Detection Mode", ["Indoor", "Outdoor"])
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_od")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = indoor_model if mode == "Indoor" else outdoor_model
        result_img, labels_text = detect_objects(image, model)
        st.image(result_img, caption="Detected Objects", use_column_width=True)

        if labels_text:
            st.write("Detected:", labels_text)
            audio_path = text_to_speech(labels_text)
            st.audio(audio_path)

# Tab 2: OCR
with tab2:
    st.header("Upload Image for OCR")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_ocr")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        text = pytesseract.image_to_string(image)
        st.subheader("Extracted Text")
        st.text_area("Text", text, height=200)

        if text.strip():
            audio_path = text_to_speech(text)
            st.audio(audio_path)
