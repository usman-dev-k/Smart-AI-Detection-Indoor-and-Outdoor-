import streamlit as st
from PIL import Image
from utils.inference import detect_objects

st.title("🚗 YOLOv8 Object Detection App")
st.markdown("Upload an image to detect objects like cars, bikes, ambulances, and more.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        result_img = detect_objects(image)
        st.image(result_img, caption="Detected Objects", use_column_width=True)
