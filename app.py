import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ---- Load model ----
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # path to your trained model
    return model

model = load_model()

# ---- Upload ----
st.title("RetinaVision: Disease Detection using YOLOv12")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Segmentation"):
        with st.spinner("Running YOLO segmentation..."):
            results = model.predict(image)
            for r in results:
                annotated_frame = r.plot()
                st.image(annotated_frame, caption="Model Prediction", use_column_width=True)
