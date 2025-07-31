import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model_path = "models/weapon.pt"
    try:
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

st.title("🔍 Weapon Detection Model Info")

model = load_model()

if model:
    try:
        class_names = model.names  # This should return a dict
        st.subheader("📦 Detected Classes:")
        st.write(list(class_names.values()))
    except Exception as e:
        st.error(f"Couldn't extract class names: {e}")
