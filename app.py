import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import time
from PIL import Image
import pytesseract

# --- Configuration ---
os.environ['YOLO_CONFIG_DIR'] = '/tmp'  # Fix for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Model paths
MODEL_DIR = "models"
INDOOR_MODEL = os.path.join(MODEL_DIR, "indoor.pt")
OUTDOOR_MODEL = os.path.join(MODEL_DIR, "outdoor.pt")

# --- WebRTC Config (STUN only) ---
def get_rtc_config():
    """RTC configuration with public STUN servers only"""
    return {
        "iceServers": [
            # Primary STUN servers
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun.iptel.org"},
            
            # Fallback STUN servers
            {"urls": "stun:stun.services.mozilla.com:3478"},
            {"urls": "stun:stun.stunprotocol.org:3478"}
        ]
    }

# --- Audio Setup ---
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

audio_queue = queue.Queue()
audio_lock = threading.Lock()
last_spoken = {"text": "", "time": 0}

def speak_text(text):
    """Thread-safe text-to-speech with cooldown"""
    with audio_lock:
        current_time = time.time()
        if text and (text != last_spoken["text"] or current_time - last_spoken["time"] > 3):
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
                last_spoken["text"] = text
                last_spoken["time"] = current_time
            except Exception as e:
                st.error(f"Speech error: {str(e)}")

def audio_worker():
    """Background audio processor"""
    while True:
        text = audio_queue.get()
        if text is None:
            break
        speak_text(text)
        audio_queue.task_done()

audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# --- Video Processor ---
class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_type = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0
        self.frame_counter = 0
        self.frame_skip = 2  # Process every 3rd frame

    def load_model(self, model_type):
        if model_type == self.model_type and self.model is not None:
            return
        
        try:
            self.model = YOLO(INDOOR_MODEL if model_type == "indoor" else OUTDOOR_MODEL)
            self.model_type = model_type
        except Exception as e:
            st.error(f"Model load failed: {str(e)}")
            self.model = None

    def recv(self, frame):
        if self.model is None:
            return frame
        
        img = frame.to_ndarray(format="bgr24")
        
        # Frame skipping for performance
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
        
        # Audio feedback
        current_time = time.time()
        if detected_objects and current_time - self.last_detection_time > self.detection_cooldown:
            audio_queue.put(f"Detected: {', '.join(detected_objects)}")
            self.last_detection_time = current_time
        
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
                audio_queue.put(text)
            else:
                st.warning("No text detected")
                
        except Exception as e:
            st.error(f"OCR failed: {str(e)}")

# --- Main App ---
def main():
    st.title("Real-Time Object Detection + OCR")
    
    tab1, tab2 = st.tabs(["Object Detection", "OCR"])
    
    with tab1:
        st.header("Object Detection")
        model_type = st.radio("Model:", ("indoor", "outdoor"), horizontal=True)
        
        # Connection advice
        with st.expander("ℹ️ Connection Tips"):
            st.write("""
            - Works best on Chrome/Firefox
            - Allow camera permissions
            - If connection fails, try:
              - Refresh the page
              - Switch networks (try mobile data)
              - Disable VPN if using one
            """)
        
        webrtc_ctx = webrtc_streamer(
            key=f"detection-{model_type}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_rtc_config(),
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            video_processor_factory=ObjectDetectionProcessor,
            async_processing=True
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.load_model(model_type)
        
        if st.button("Reconnect Camera"):
            st.experimental_rerun()
    
    with tab2:
        ocr_to_speech()
    
    # Cleanup
    if not st.session_state.get('_is_running', False):
        audio_queue.put(None)
        audio_thread.join(timeout=1.0)
        st.session_state._is_running = True

if __name__ == "__main__":
    main()
