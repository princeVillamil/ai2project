import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="RetinaVision YOLOv12", layout="centered")
st.title("RetinaVision: Pathological Myopia Detection")

# --- Load YOLO model ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # path to your trained model
    return model

model = load_model()

# --- Upload section ---
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Run YOLO inference ---
    with st.spinner("Analyzing image..."):
        results = model(img_np)

    # --- Process results ---
    r = results[0]
    img = img_bgr.copy()
    info_text = []

    # Overlay segmentation masks
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        for mask in masks:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8) * 255
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 2] = mask  # red channel
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

    # Draw boxes and info
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        label = f"{names[cls_id]} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_text.append(
            f"Class: **{names[cls_id]}** | Confidence: `{score:.2f}` | Area: `{area}` | Box: `{x2-x1}x{y2-y1}`"
        )

    # --- Display results ---
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Prediction Result", use_container_width=True)
    st.markdown("---")

    st.markdown("### Detection Info")
    if info_text:
        for info in info_text:
            st.markdown(info)
    else:
        st.info("No detections found.")
