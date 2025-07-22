import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract
import pyttsx3
import threading
import logging

# Disable noisy ALSA warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# YOLO model paths
MODEL_DIR = "models"
INDOOR_MODEL = os.path.join(MODEL_DIR, "indoor.pt")
OUTDOOR_MODEL = os.path.join(MODEL_DIR, "outdoor.pt")

# RTC Configuration with TURN fallback
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:global.stun.twilio.com:3478"},
        {
            "urls": "turn:global.turn.twilio.com:3478?transport=udp",
            "username": "b9e6f8ff9be8b7303e3520570113cff848385c3c60b83b17adaab2e5a607385c",  # <-- Your Twilio SID
            "credential": "ZT8h0y7ShKOWLmtyYH845iay2/w+0i0GNFVwZ73/1qw="  # <-- Your Twilio Auth Token
        }
    ]
})

# Text-to-speech engine (runs in a thread to avoid blocking)
def speak_text(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
    threading.Thread(target=_speak, daemon=True).start()

# Object detection video processor
class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_type = None
        self.frame_counter = 0
        self.frame_skip = 2
        self.last_detected = set()

    def load_model(self, model_type):
        if self.model_type != model_type:
            try:
                path = INDOOR_MODEL if model_type == "indoor" else OUTDOOR_MODEL
                self.model = YOLO(path)
                self.model_type = model_type
                logger.info(f"{model_type} model loaded.")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.model = None

    def recv(self, frame):
        if self.model is None:
            return frame

        img = frame.to_ndarray(format="bgr24")

        self.frame_counter = (self.frame_counter + 1) % (self.frame_skip + 1)
        if self.frame_counter != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        results = self.model.predict(img, persist=True, verbose=False)
        detected_now = set()

        for result in results:
            if result.boxes is not None:
                for box, conf, cls_id in zip(result.boxes.xyxy.cpu().numpy(),
                                             result.boxes.conf.cpu().numpy(),
                                             result.boxes.cls.cpu().numpy().astype(int)):
                    if conf > 0.5:
                        label = self.model.names[cls_id]
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected_now.add(label)

        new_objects = detected_now - self.last_detected
        if new_objects:
            for obj in new_objects:
                speak_text(f"{obj} detected")

        self.last_detected = detected_now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# OCR & TTS
def ocr_to_speech():
    st.subheader("Capture Image for OCR")
    img_file = st.camera_input("Take a photo")

    if img_file:
        try:
            img = Image.open(img_file)
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            st.write("Extracted Text:")
            st.success(text.strip() if text.strip() else "No readable text found")

            if text.strip():
                speak_text(text.strip())

        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            st.error("Failed to process image.")

# Main UI
def main():
    st.set_page_config(page_title="Object Detection & OCR App")
    st.title("📸 Real-Time Object Detection & OCR to Speech")

    tab1, tab2 = st.tabs(["🎯 Object Detection", "📝 OCR to Speech"])

    with tab1:
        st.header("Select Detection Mode")
        model_type = st.radio("Choose Model", ("indoor", "outdoor"), horizontal=True)

        webrtc_ctx = webrtc_streamer(
            key=f"det-{model_type}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            video_processor_factory=ObjectDetectionProcessor,
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.load_model(model_type)

    with tab2:
        ocr_to_speech()

if __name__ == "__main__":
    main()
