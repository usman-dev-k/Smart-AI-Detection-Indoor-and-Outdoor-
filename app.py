import streamlit as st
import cv2
import numpy as np
import pytesseract
import pyttsx3
import threading
import time
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

# Initialize TTS engine once
if 'tts_engine' not in st.session_state:
    st.session_state.tts_engine = pyttsx3.init()
    st.session_state.tts_engine.setProperty('rate', 150)

# Load models with caching
@st.cache_resource
def load_model(model_path):
    return attempt_load(model_path, map_location='cpu')

def detect_objects(model, frame, conf_thres=0.5):
    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, 0.45)
    
    # Process detections
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                detections.append((xyxy, label.split()[0], conf))  # Store only object name
    
    return detections

def speak(text):
    st.session_state.tts_engine.say(text)
    st.session_state.tts_engine.runAndWait()

def main():
    st.title("Real-Time Vision Assistant")
    
    tab1, tab2 = st.tabs(["Live Object Detection", "Instant OCR Reader"])
    
    # TAB 1: Real-Time Object Detection
    with tab1:
        st.header("Real-Time Object Detection with Audio Feedback")
        
        model_choice = st.radio("Select Model:", ("Indoor", "Outdoor"), horizontal=True)
        confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
        
        start_btn = st.button("Start Camera")
        stop_btn = st.button("Stop Camera")
        
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        if start_btn:
            st.session_state.camera_active = True
            model_path = f"models/{model_choice.lower()}.pt"
            model = load_model(model_path)
            
            video_placeholder = st.empty()
            cap = cv2.VideoCapture(0)
            
            last_spoken = time.time()
            cooldown = 2  # seconds
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                # Detection
                detections, frame = detect_objects(model, frame, confidence), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw boxes and collect objects
                detected_items = set()
                for (x1, y1, x2, y2), name, conf in detections:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_items.add(name)
                
                # Show frame
                video_placeholder.image(frame, channels="RGB")
                
                # Audio feedback
                if detected_items and (time.time() - last_spoken) > cooldown:
                    objects_str = ", ".join(detected_items)
                    threading.Thread(target=speak, args=(f"I see {objects_str}",)).start()
                    last_spoken = time.time()
                
                if stop_btn:
                    st.session_state.camera_active = False
                    break
            
            cap.release()
            if not st.session_state.camera_active:
                video_placeholder.empty()
    
    # TAB 2: OCR Reader
    with tab2:
        st.header("Instant OCR to Speech")
        
        col1, col2 = st.columns(2)
        with col1:
            capture_btn = st.button("Capture Text")
        with col2:
            speak_btn = st.button("Speak Text")
        
        if 'captured_image' not in st.session_state:
            st.session_state.captured_image = None
            st.session_state.extracted_text = ""
        
        if capture_btn:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                st.session_state.captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Preprocess for better OCR
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(thresh)
                st.session_state.extracted_text = text.strip()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.captured_image is not None:
                    st.image(st.session_state.captured_image, caption="Captured Image")
            with col2:
                st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
        
        if speak_btn and st.session_state.extracted_text:
            threading.Thread(target=speak, args=(st.session_state.extracted_text,)).start()
            st.toast("Speaking text...")

if __name__ == "__main__":
    main()
