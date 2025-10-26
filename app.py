from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

# Load trained YOLO model
model = YOLO("best.pt")  # Change path if needed

# Ask the user for an image file path
img_path = input("Enter the path to your image file: ").strip()

# Check if file exists
if not os.path.exists(img_path):
    print("File not found. Please check the path and try again.")
    exit()

# Run inference
results = model(img_path)

# Process and visualize results
for r in results:
    # Load the original image (full resolution)
    img = cv2.imread(img_path)

    # Overlay segmentation masks
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        for mask in masks:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(np.uint8) * 255
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 2] = mask  # Red overlay for segmentation
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)   

    # Draw bounding boxes, labels, and confidence scores
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]} {score:.2f}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Display image at original resolution
    h, w = img.shape[:2]
    dpi = 300
    fig_w, fig_h = w / dpi, h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    plt.show()

    # Print confidence info
    print("\n--- Detection Info ---")
    for box, score, cls_id in zip(boxes, scores, class_ids):
        print(f"Class: {names[cls_id]}, Confidence: {score:.2f}")
