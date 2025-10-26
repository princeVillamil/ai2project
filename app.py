import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your trained model
model = YOLO("best.pt")

st.title("RetinaVision Object Segmentation")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Running model..."):
            # Run inference
            results = model.predict(source=np.array(image), conf=0.25)
            st.success("Prediction complete!")

            # Process the first result
            for r in results:
                img = np.array(image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Overlay segmentation masks
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    for mask in masks:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask = mask.astype(np.uint8) * 255
                        colored_mask = np.zeros_like(img)
                        colored_mask[:, :, 2] = mask  # Red overlay
                        img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

                # Draw bounding boxes, labels, confidence
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = model.names

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls_id]} {score:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                lineType=cv2.LINE_AA)

                # Convert to RGB for Streamlit display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Detection Result", use_column_width=True)
