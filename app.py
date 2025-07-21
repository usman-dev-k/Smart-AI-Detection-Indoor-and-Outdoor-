import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
from TTS.api import TTS
import pytesseract
import tempfile
import base64
import os

# ========== Load TTS Model ==========
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# ========== Load YOLO Models ==========
@st.cache_resource
def load_models():
    indoor = YOLO("models/indoor.pt")
    outdoor = YOLO("models/outdoor.pt")
    return indoor, outdoor

indoor_model, outdoor_model = load_models()

OUTDOOR_CLASS_NAMES = [
    'Ambulance', 'Auto-Rikshaw', 'bike', 'bus', 'car',
    'puddle', 'stairs', 'truck', 'van', 'zebra-crossing'
]

# ========== Image Preprocessing for OCR ==========
def preprocess_image(pil_image):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    pil_thresh = Image.fromarray(thresh)
    enhancer = ImageEnhance.Contrast(pil_thresh)
    return enhancer.enhance(2.0)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Assistive Vision", layout="wide")
st.title("🧠 Assistive Vision System")

mode = st.sidebar.selectbox("Choose Mode", ["🧍 Object Detection", "🔠 OCR to TTS"])

# ========== Object Detection ==========
if mode == "🧍 Object Detection":
    env = st.radio("Select Environment", ["Indoor", "Outdoor"])
    model = indoor_model if env == "Indoor" else outdoor_model
    class_names = model.names if env == "Indoor" else OUTDOOR_CLASS_NAMES
    st.info(f"Running object detection in {env} environment")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_sentence = ""

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(source=img, conf=0.4)[0]

            labels = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id] if cls_id < len(class_names) else "Unknown"
                labels.append(label)
                xyxy = box.xyxy[0].int().tolist()
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if labels:
                sentence = " and ".join(set(labels)) + " ahead"
                if sentence != self.last_sentence:
                    self.last_sentence = sentence
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tts_model.tts_to_file(text=sentence, file_path=tmpfile.name)
                        tmpfile.seek(0)
                        b64 = base64.b64encode(tmpfile.read()).decode()
                        audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ========== OCR to TTS ==========
elif mode == "🔠 OCR to TTS":
    st.subheader("📷 Capture Image for OCR")
    img = st.camera_input("Take a picture")

    if img:
        image = Image.open(img)
        st.image(image, caption="Captured Image", use_column_width=True)

        proc_img = preprocess_image(image)
        st.image(proc_img, caption="Preprocessed Image", use_column_width=True)

        text = pytesseract.image_to_string(proc_img)
        text = " ".join(text.split())

        if text:
            st.subheader("📝 Extracted Text")
            st.success(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tts_model.tts_to_file(text=text, file_path=tmpfile.name)
                tmpfile.seek(0)
                b64 = base64.b64encode(tmpfile.read()).decode()
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
                """
                st.subheader("🗣 Speaking...")
                st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("No text found.")
