import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import pytesseract
from PIL import Image

# Load models
indoor_model = YOLO("models/indoor.pt")  # replace with actual path
outdoor_model = YOLO("models/outdoor.pt")  # replace with actual path

# Utility: Convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Object detection function
def detect_objects(image, scene):
    model = indoor_model if scene == "Indoor" else outdoor_model
    results = model(image)
    boxes = results[0].boxes
    names = results[0].names
    spoken_text = []

    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        spoken_text.append(label)

        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    speech = text_to_speech(", ".join(spoken_text)) if spoken_text else None
    return image, speech

# OCR function
def ocr_from_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    text = pytesseract.image_to_string(pil_img)
    audio = text_to_speech(text) if text.strip() else None
    return text, audio

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 Smart Vision App")
    with gr.Tab("📦 Object Detection"):
        scene = gr.Radio(["Indoor", "Outdoor"], value="Indoor", label="Scene Type")
        webcam = gr.Image(source="webcam", streaming=True, label="Webcam Feed")
        out_img = gr.Image(label="Detected Objects")
        out_audio = gr.Audio(label="TTS Output", interactive=False, type="filepath")
        btn = gr.Button("Detect")
        btn.click(fn=detect_objects, inputs=[webcam, scene], outputs=[out_img, out_audio])

    with gr.Tab("📖 OCR Text to Speech"):
        webcam2 = gr.Image(source="webcam", streaming=False, label="Take Picture")
        ocr_text = gr.Textbox(label="Detected Text")
        ocr_audio = gr.Audio(label="TTS Output", interactive=False, type="filepath")
        btn2 = gr.Button("Read Text")
        btn2.click(fn=ocr_from_image, inputs=webcam2, outputs=[ocr_text, ocr_audio])

# Launch
demo.launch()
