import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import easyocr
import tempfile
import base64
from gtts import gTTS
import os


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    indoor_model = YOLO("models/indoor.pt")
    outdoor_model = YOLO("models/outdoor.pt")
    return indoor_model, outdoor_model


indoor_model, outdoor_model = load_models()

OUTDOOR_CLASS_NAMES = [
    'Ambulance', 'Auto-Rikshaw', 'bike', 'bus', 'car',
    'puddle', 'stairs', 'truck', 'van', 'zebra-crossing'
]


# =========================
# OCR PREPROCESSING
# =========================
def preprocess_image(pil_image):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    pil_thresh = Image.fromarray(thresh)
    enhancer = ImageEnhance.Contrast(pil_thresh)
    enhanced_image = enhancer.enhance(2.0)

    return enhanced_image


# =========================
# AUDIO PLAYER (NO AUTOPLAY)
# =========================
def show_audio_bar(text):
    tts = gTTS(text=text, lang='en')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()
    os.remove(audio_path)

    audio_html = f"""
    <audio controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Assistive Vision System", layout="wide")
st.title("🧠 Assistive Vision System")


app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["🧍 Object Detection", "🔠 OCR to TTS"]
)


# ============================================================
# OBJECT DETECTION — IMAGE ONLY
# ============================================================
if app_mode == "🧍 Object Detection":

    env = st.radio("Select Environment:", ["Indoor", "Outdoor"])
    model = indoor_model if env == "Indoor" else outdoor_model
    class_names = model.names if env == "Indoor" else OUTDOOR_CLASS_NAMES

    st.info("Upload or capture an image for object detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    captured = st.camera_input("Or take a picture")

    image_source = uploaded if uploaded else captured

    if image_source:
        image = Image.open(image_source).convert("RGB")
        img_np = np.array(image)

        results = model.predict(source=img_np, conf=0.4)[0]

        detected = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])

            try:
                label = class_names[cls_id]
            except IndexError:
                continue

            detected[label] = detected.get(label, 0) + 1

            xyxy = box.xyxy[0].int().tolist()

            cv2.rectangle(
                img_np,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img_np,
                label,
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        st.image(img_np, caption="Detected Objects", use_column_width=True)

        if detected:
            parts = []
            for label, count in detected.items():
                parts.append(
                    f"{count} {label}" if count == 1 else f"{count} {label}s"
                )

            sentence = " and ".join(parts) + " ahead"

            st.success(sentence)
            show_audio_bar(sentence)
        else:
            st.warning("No objects detected.")


# ============================================================
# OCR → TEXT → SPEECH
# ============================================================
elif app_mode == "🔠 OCR to TTS":

    st.subheader("📷 Capture or Upload Image for OCR")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    captured = st.camera_input("Or take a picture")

    image_source = uploaded if uploaded else captured

    if image_source:
        image = Image.open(image_source).convert("RGB")

        st.image(image, caption="Original Image", use_column_width=True)

        processed = preprocess_image(image)
        st.image(processed, caption="Preprocessed Image", use_column_width=True)

        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(np.array(processed))

        text = " ".join([res[1] for res in results])

        if text:
            st.subheader("📝 Extracted Text")
            st.success(text)
            show_audio_bar(text)
        else:
            st.warning("No text found in the image.")
