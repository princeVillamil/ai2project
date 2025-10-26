import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import os
import numpy as np

# Title and instructions
st.title("RetinaVision - Eye Disease Segmentation")
st.write("Upload an eye image to segment AMD, Cataract, and Pathological Myopia.")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Confidence threshold slider
    conf = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

    # Run inference
    with st.spinner("Running segmentation..."):
        # Save uploaded image to a temporary file with proper format
        ext = os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            image.save(tmp_file.name, format=image.format or "PNG")
            results = model.predict(tmp_file.name, conf=conf, imgsz=800)

        # Create a copy for drawing
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Draw bounding boxes if available
        if hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw masks if available
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for mask_tensor in results[0].masks.data:
                mask = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask).convert("L")
                result_image.paste(mask_image, (0, 0), mask_image)

        # Display result
        st.image(result_image, caption="Segmented Output", use_container_width=True)

        # Detection details
        st.subheader("Detection Details")
        st.json(results[0].tojson())
