import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="Nail Disease Segmentation", layout="wide")

# --- Configuration ---
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.20
MASK_ALPHA = 0.5

# --- Inject Custom CSS ---
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.stApp > header {
    display: none;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.main .block-container {
    padding: 2rem;
    max-width: 1200px;
    margin: auto;
    background-color: #e8edf5;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Left Panel (White - Upload Area) */
div[data-testid="stVerticalBlock"]:nth-child(1) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background-color: white;
    padding: 3rem;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    height: 100%;
    box-sizing: border-box;
}
div[data-testid="stVerticalBlock"]:nth-child(1) h1,
div[data-testid="stVerticalBlock"]:nth-child(1) h2,
div[data-testid="stVerticalBlock"]:nth-child(1) h3 {
    color: #333 !important;
    padding-bottom: 1rem;
}
div[data-testid="stVerticalBlock"]:nth-child(1) div,
div[data-testid="stVerticalBlock"]:nth-child(1) p,
div[data-testid="stVerticalBlock"]:nth-child(1) li {
    color: #555 !important;
}

/* Right Panel (Blue - Info) */
div[data-testid="stVerticalBlock"]:nth-child(2) > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
    background-color: #0047AB;
    color: white;
    padding: 3rem;
    border-radius: 10px;
    height: 100%;
    box-sizing: border-box;
}
div[data-testid="stVerticalBlock"]:nth-child(2) * {
    color: white !important;
}
div[data-testid="stVerticalBlock"]:nth-child(2) h1,
div[data-testid="stVerticalBlock"]:nth-child(2) h2,
div[data-testid="stVerticalBlock"]:nth-child(2) h3 {
    color: white !important;
    padding-bottom: 1rem;
}

/* File uploader */
div[data-testid="stFileUploader"] section {
    border: 2px dashed #ccc;
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 5px;
}
div[data-testid="stFileUploader"] section:hover {
    background-color: #e9ecef;
}

/* Spinner text */
.stSpinner > div > div {
    color: #0047AB !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Define Columns (reversed order) ---
col_left, col_right = st.columns([1.2, 1], gap="large")

# --- LEFT COLUMN (White) ---
with col_left:
    st.subheader("Analyze Your Image")

    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.error(f"FATAL ERROR: Model failed to load from path '{MODEL_PATH}'. Ensure 'best.pt' is present.")
        st.stop()
    else:
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            result_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                img_cv = np.array(image.convert('RGB'))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                result_placeholder.image(image, caption='Uploaded Image.', use_container_width=True)

                with st.spinner("Analyzing image..."):
                    results = model(img_cv, conf=CONFIDENCE_THRESHOLD)

                    overlay_image = img_cv.copy()
                    detection_made = False
                    detected_classes = set()

                    names = model.names
                    colors = [tuple(np.random.randint(100, 256, 3).tolist()) for _ in range(len(names))]

                    for r in results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()

                        if boxes.shape[0] > 0:
                            class_ids = r.boxes.cls.cpu().numpy().astype(int)
                            detection_made = True
                            for cls_id in class_ids:
                                if 0 <= cls_id < len(names):
                                    detected_classes.add(names[cls_id])
                        else:
                            class_ids = np.array([], dtype=int)

                        # Segmentation Masks
                        if r.masks is not None and len(class_ids) > 0:
                            masks = r.masks.data.cpu().numpy()
                            overlay_h, overlay_w = overlay_image.shape[:2]

                            for i, mask in enumerate(masks):
                                if i < len(class_ids):
                                    mask_resized = cv2.resize(mask, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)
                                    mask_uint8 = mask_resized.astype(np.uint8) * 255
                                    class_id = class_ids[i]
                                    if 0 <= class_id < len(colors):
                                        color = colors[class_id]
                                        colored_mask = np.zeros_like(overlay_image, dtype=np.uint8)
                                        for c_idx in range(3):
                                            colored_mask[:, :, c_idx] = np.where(mask_uint8 == 255, color[c_idx], 0)
                                        overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask, MASK_ALPHA, 0)

                        # Bounding Boxes
                        if class_ids.size > 0:
                            for i, (box, score) in enumerate(zip(boxes, scores)):
                                if i < len(class_ids):
                                    cls_id = class_ids[i]
                                    if 0 <= cls_id < len(names):
                                        x1, y1, x2, y2 = map(int, box)
                                        label = f"{names[cls_id]} {score:.2f}"
                                        color = colors[cls_id]
                                        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
                                        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                        y_text_bg_top = max(0, y1 - label_height - baseline - 5)
                                        cv2.rectangle(overlay_image, (x1, y_text_bg_top), (x1 + label_width, y1), color, cv2.FILLED)
                                        y_text_pos = max(10, y1 - baseline - 3)
                                        cv2.putText(overlay_image, label, (x1, y_text_pos),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                                                    lineType=cv2.LINE_AA)

                # Display Results
                if detection_made:
                    is_only_healthy = detected_classes == {'healthy_nail'}
                    print(detected_classes)
                    result_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                    if is_only_healthy:
                        message_placeholder.success("Healthy nail detected. No diseases found.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image - Healthy Nail.', use_container_width=True)
                    else:
                        message_placeholder.warning("Nail condition(s) detected. Please consult a healthcare professional.")
                        result_placeholder.image(result_image_rgb, caption='Processed Image with Detections.', use_container_width=True)
                else:
                    message_placeholder.info(f"No nail conditions detected above {CONFIDENCE_THRESHOLD*100:.0f}% confidence.")
                    result_placeholder.image(image, caption='Uploaded Image.', use_container_width=True)

            except Exception as e:
                result_placeholder.empty()
                message_placeholder.empty()
                st.error(f"An error occurred during image processing: {e}")
                st.warning("Please upload a valid, uncorrupted image file.")

# --- RIGHT COLUMN (Blue Info Panel) ---
with col_right:
    st.header("Eye Disease Segmentation")
    st.write(f"""
        This project introduces a deep learning–based system for the detection and segmentation of ocular diseases using retinal fundus images. Leveraging the YOLOv12 segmentation architecture, the model was trained on a curated dataset called RetinaVision, consisting of annotated images representing cataract, age-related macular degeneration (AMD), and pathologic myopia. The dataset was divided into 80% training, 10% validation, and 10% testing subsets to ensure balanced evaluation.
    """)
    st.markdown("---")


