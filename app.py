import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Load your YOLO model once
@st.cache_resource
def load_model():
    model_path = "/content/drive/MyDrive/RetinaVision_YOLOv12/yolov12/runs/segment/train_newDataSet_Fri_2025-10-24_03-33/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# Streamlit UI
st.title("🧠 RetinaVision – Image Segmentation & Detection")
st.write("Upload an image to detect and visualize segmentation masks with YOLOv12.")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Run prediction button
    if st.button("🔍 Run Prediction"):
        with st.spinner("Running model..."):
            # Convert PIL → NumPy (for YOLO)
            img_np = np.array(image)
            results = model(img_np)

        st.success("✅ Prediction complete!")

        # Process and visualize results
        for r in results:
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Apply segmentation masks
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for mask in masks:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(np.uint8) * 255
                    colored_mask = np.zeros_like(img)
                    colored_mask[:, :, 2] = mask  # Red overlay
                    img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

            # Draw boxes and labels
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names[cls_id]} {score:.2f}"

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            lineType=cv2.LINE_AA)

            # Convert BGR → RGB for Streamlit
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display full-resolution image with matplotlib (sharp edges)
            h, w = img_rgb.shape[:2]
            dpi = 300
            fig_w, fig_h = w / dpi, h / dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.imshow(img_rgb, interpolation='nearest')
            ax.axis("off")
            st.pyplot(fig)

            # Optional: Show detection info
            st.subheader("📊 Detection Summary")
            for box, score, cls_id in zip(boxes, scores, class_ids):
                st.write(f"**Class:** {names[cls_id]} — **Confidence:** {score:.2f}")

