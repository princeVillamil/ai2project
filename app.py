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
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---- Run model ----
    st.write("Running detection...")
    results = model.predict(image_bgr, conf=0.5, imgsz=800)

    # ---- Display predictions ----
    for r in results:
        annotated_frame = r.plot()  # YOLO automatically draws boxes and masks
        st.image(annotated_frame, caption="Detection Result", use_container_width=True)
