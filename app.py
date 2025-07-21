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

# Fix Ultralytics config directory issue
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

# Set Tesseract path (adjust for your environment)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Constants
MODEL_DIR = "models"
INDOOR_MODEL = os.path.join(MODEL_DIR, "indoor.pt")
OUTDOOR_MODEL = os.path.join(MODEL_DIR, "outdoor.pt")

# Check if models exist
if not os.path.exists(INDOOR_MODEL):
    st.error(f"Indoor model not found at {INDOOR_MODEL}")
if not os.path.exists(OUTDOOR_MODEL):
    st.error(f"Outdoor model not found at {OUTDOOR_MODEL}")

# Initialize text-to-speech engine with fallback
tts_engine = None
try:
    tts_engine = pyttsx3.init()
    
    # Handle voice initialization errors
    try:
        tts_engine.setProperty('rate', 150)
    except Exception as e:
        st.warning(f"Couldn't set speech rate: {str(e)}")
    
    # Try to set a working voice
    voices = tts_engine.getProperty('voices')
    if voices:
        try:
            tts_engine.setProperty('voice', voices[0].id)
        except:
            pass  # Use default voice if specific selection fails
except Exception as e:
    st.error(f"Text-to-speech initialization failed: {str(e)}")
    st.warning("Audio feedback will be disabled")

# Audio feedback queue and lock
audio_queue = queue.Queue()
audio_lock = threading.Lock()
last_spoken = {"text": "", "time": 0}

def speak_text(text):
    """Thread-safe text-to-speech with cooldown"""
    if tts_engine is None:
        return
    
    with audio_lock:
        current_time = time.time()
        # Prevent repeating the same text within 3 seconds
        if text and (text != last_spoken["text"] or current_time - last_spoken["time"] > 3):
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
                last_spoken["text"] = text
                last_spoken["time"] = current_time
            except Exception as e:
                st.error(f"Speech synthesis error: {str(e)}")

def audio_worker():
    """Background worker for audio processing"""
    while True:
        text = audio_queue.get()
        if text is None:  # Termination signal
            break
        speak_text(text)
        audio_queue.task_done()

# Start audio thread
audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_type = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        self.last_frame_time = time.time()
        self.frame_skip = 2  # Process every 3rd frame

    def load_model(self, model_type):
        """Load appropriate YOLO model with caching"""
        if model_type == self.model_type and self.model is not None:
            return
        
        try:
            if model_type == "indoor":
                self.model = YOLO(INDOOR_MODEL)
            else:
                self.model = YOLO(OUTDOOR_MODEL)
            self.model_type = model_type
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            self.model = None

    def recv(self, frame):
        """Process each video frame"""
        current_time = time.time()
        
        # Skip frames for performance
        if current_time - self.last_frame_time < 0.1:  # 10ms = ~100 FPS throttling
            return frame
        self.last_frame_time = current_time
        
        if self.model is None:
            return frame
        
        img = frame.to_ndarray(format="bgr24")
        
        # Process every nth frame (frame skipping)
        self.frame_counter = (self.frame_counter + 1) % self.frame_skip
        if self.frame_counter != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        results = self.model.track(img, persist=True, verbose=False)
        
        # Process detections
        detected_objects = set()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    label = self.model.names[cls_id]
                    confidence = float(conf)
                    
                    if confidence > 0.5:
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {confidence:.2f}", 
                                    (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 255, 0), 2)
                        
                        detected_objects.add(label)
        
        # Audio feedback with cooldown
        if detected_objects and current_time - self.last_detection_time > self.detection_cooldown:
            objects_text = ", ".join(detected_objects)
            audio_text = f"Detected: {objects_text}"
            audio_queue.put(audio_text)
            self.last_detection_time = current_time
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def ocr_to_speech():
    """Capture image and convert text to speech"""
    st.write("### OCR Text-to-Speech")
    st.write("Position text in front of your camera and click Capture")
    
    img_file = st.camera_input("Capture text for OCR")
    
    if img_file is not None:
        try:
            img = Image.open(img_file)
            st.image(img, caption="Captured Image", use_column_width=True)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            st.subheader("Extracted Text:")
            st.write(text)
            
            # Text-to-speech
            if text.strip():
                audio_queue.put(text)
                st.success("Speaking extracted text...")
            else:
                st.warning("No text detected in the image")
                
        except Exception as e:
            st.error(f"OCR processing failed: {str(e)}")

def main():
    st.title("Real-Time Perception System")
    st.write("Object Detection + OCR Text-to-Speech")
    
    tab1, tab2 = st.tabs(["Object Detection", "OCR Text-to-Speech"])
    
    with tab1:
        st.header("Real-Time Object Detection")
        model_type = st.radio("Select Model:", ("outdoor", "indoor"), horizontal=True)
        
        # Enhanced STUN/TURN configuration
        rtc_config = {
            "iceServers": [
                # Primary STUN (Google)
                {"urls": "stun:stun.l.google.com:19302"},
                
                # Backup STUN servers
                {"urls": "stun:stun1.l.google.com:19302"},
                {"urls": "stun:stun2.l.google.com:19302"},
                {"urls": "stun:stun.iptel.org"},
                
                # Free TURN server (fallback)
                # {
                #     # "urls": "turn:numb.viagenie.ca",
                #     # "username": "your-email@gmail.com",
                #     # "credential": "your-password"
                # }
            ]
        }
        
        # Add a connection troubleshooting section
        with st.expander("Connection Troubleshooting"):
            st.markdown("""
            **If the camera doesn't load:**
            1. Refresh the page and allow camera permissions
            2. Check your network firewall settings
            3. Try a different browser (Chrome/Firefox work best)
            4. Use a mobile hotspot if corporate network blocks STUN
            """)
        
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},  # Lower resolution for mobile
                        "height": {"ideal": 480}
                    },
                    "audio": False
                },
                video_processor_factory=ObjectDetectionProcessor,
                async_processing=True,
            )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.load_model(model_type)
            webrtc_ctx.video_processor.frame_counter = 0
            
        st.info("""
        **Mobile Instructions:**
        1. Allow camera access when prompted
        2. Point camera at objects
        3. Audio feedback will announce detections
        """)
        
        # Add a reset button for connection issues
        if st.button("Reconnect Camera"):
            st.experimental_rerun()
    
    with tab2:
        ocr_to_speech()
    
    # Cleanup on app exit
    if not st.session_state.get('_is_running', False):
        audio_queue.put(None)
        audio_thread.join(timeout=1.0)
        st.session_state._is_running = True

if __name__ == "__main__":
    main()
