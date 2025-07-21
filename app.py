import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract
import logging

# Disable noisy ALSA warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Model paths
MODEL_DIR = "models"
INDOOR_MODEL = os.path.join(MODEL_DIR, "indoor.pt")
OUTDOOR_MODEL = os.path.join(MODEL_DIR, "outdoor.pt")

# --- WebRTC Config ---
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        # Primary STUN (IPv4)
        {"urls": "stun:74.125.250.129:19302"},
        
        # Fallback STUN servers
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        
        # TURN fallback (if you add credentials later)
        # {
        #     "urls": "turn:your-turn-server.com:3478",
        #     "username": "username",
        #     "credential": "password"
        # }
    ]
})

# --- Video Processor ---
class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_type = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0
        self.frame_counter = 0
        self.frame_skip = 2

    def load_model(self, model_type):
        if model_type == self.model_type and self.model is not None:
            return
        
        try:
            self.model = YOLO(INDOOR_MODEL if model_type == "indoor" else OUTDOOR_MODEL)
            self.model_type = model_type
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")
            self.model = None

    def recv(self, frame):
        if self.model is None:
            return frame
        
        img = frame.to_ndarray(format="bgr24")
        
        # Frame skipping
        self.frame_counter = (self.frame_counter + 1) % (self.frame_skip + 1)
        if self.frame_counter != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Detection
        results = self.model.track(img, persist=True, verbose=False)
        detected_objects = set()
        
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
                        detected_objects.add(label)
        
        # Visual feedback only (no audio)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- OCR Function ---
def ocr_to_speech():
    st.write("### OCR Text-to-Speech")
    img_file = st.camera_input("Capture text for OCR")
    
    if img_file:
        try:
            img = Image.open(img_file)
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh)
            
            if text.strip():
                st.write("Extracted Text:", text)
                # Visual feedback only (no audio)
            else:
                st.warning("No text detected")
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")

# --- Main App ---
def main():
    st.title("Real-Time Object Detection + OCR")
    
    tab1, tab2 = st.tabs(["Object Detection", "OCR"])
    
    with tab1:
        st.header("Object Detection")
        model_type = st.radio("Model:", ("indoor", "outdoor"), horizontal=True)
        
        webrtc_ctx = webrtc_streamer(
            key=f"detection-{model_type}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False  # Disable audio completely
            },
            video_processor_factory=ObjectDetectionProcessor,
            async_processing=True
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.load_model(model_type)
    
    with tab2:
        ocr_to_speech()

if __name__ == "__main__":
    main()
