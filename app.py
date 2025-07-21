import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
from ultralytics import YOLO
import tempfile
from streamlit_webrtc import WebRtcMode
import av

st.set_page_config(layout="wide")
st.title("📷 Smart Camera App")

# Load models
indoor_model = YOLO("models/indoor.pt")
outdoor_model = YOLO("models/outdoor.pt")

# Utility: TTS
def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# -------- Object Detection Webcam --------
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.labels_spoken = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        annotated = results[0].plot()

        # Get spoken labels
        labels = results[0].names
        spoken = set()
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            spoken.add(labels[class_id])
        self.labels_spoken = list(spoken)
        return annotated

# -------- OCR Snapshot --------
def process_ocr_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# ---------------- UI -------------------
tab1, tab2 = st.tabs(["📦 Real-time Object Detection", "🔎 OCR from Camera"])

# ---------- TAB 1: Real-Time Object Detection ----------
with tab1:
    st.subheader("Real-time Object Detection via Webcam")
    model_choice = st.radio("Choose model", ["Indoor", "Outdoor"], horizontal=True)
    selected_model = indoor_model if model_choice == "Indoor" else outdoor_model

    RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
    
    ctx = webrtc_streamer(
        key="realtime-od",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: ObjectDetectionTransformer(selected_model),
        async_processing=True
    )

    if ctx.video_transformer:
        spoken_labels = ctx.video_transformer.labels_spoken
        if spoken_labels:
            st.markdown("### Detected:")
            label_str = ", ".join(spoken_labels)
            st.write(label_str)
            audio_path = text_to_speech(label_str)
            st.audio(audio_path)

# ---------- TAB 2: OCR ----------
with tab2:
    st.subheader("OCR from Camera Snapshot")

    image = st.camera_input("Take a photo")

    if image:
        img = Image.open(image)
        img_np = np.array(img)
        text = process_ocr_image(img_np)

        st.markdown("### Extracted Text:")
        st.text_area("Text", text, height=200)

        if text.strip():
            audio_path = text_to_speech(text)
            st.audio(audio_path)
