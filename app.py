import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np

# -------------------------------
# Load YOLO model
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # your trained model file
    return model

model = load_model()

# -------------------------------
# Streamlit App
# -------------------------------
st.title("YOLOv12 Segmentation Viewer")
st.write("Upload an image to visualize detections and segmentation masks.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running model..."):
            results = model.predict(source=np.array(image), conf=0.25)

        st.success("Prediction complete!")

        for r in results:
            annotated_image = r.plot()  # annotated numpy image
            st.image(annotated_image, caption="Predicted Output", use_column_width=True)

            # Detection details
            boxes = r.boxes
            names = model.names

            if boxes is not None and len(boxes) > 0:
                data = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    data.append({
                        "Class": cls_name,
                        "Confidence": round(conf, 3),
                        "X1": round(xyxy[0], 1),
                        "Y1": round(xyxy[1], 1),
                        "X2": round(xyxy[2], 1),
                        "Y2": round(xyxy[3], 1),
                    })

                st.subheader("Detection Details")
                st.dataframe(pd.DataFrame(data))
            else:
                st.warning("No detections found.")
