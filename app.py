import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Title and instructions
st.title("RetinaVision - Eye Disease Segmentation")
st.write("Upload an eye image to segment AMD, Cataract, and Pathological Myopia.")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # path to your trained model
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Confidence threshold slider
    conf = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

    # Run inference
    with st.spinner("Running segmentation..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            image.save(tmp_file.name)
            results = model.predict(tmp_file.name, conf=conf, imgsz=800)

        # Display result
        result_image = results[0].plot()  # annotated output
        st.image(result_image, caption="Segmented Output", use_container_width=True)

        # Detection details
        st.subheader("Detection Details")
        st.json(results[0].tojson())
