import os
import time
from typing import Optional
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import cv2

# --- Global memory for frame skipping ---
_frame_counter = 0
_last_processed_frame = None
_last_metrics = "Waiting for first frame..."
_last_label = "Normal Driving"

CLASS_LABELS = {
    0: "Normal Driving",
    1: "Operating the Radio",
    2: "Reaching Behind",
    3: "Eating",
    4: "Drinking",
    5: "Using Phone",
    6: "Taking Phone",
    7: "Eyes Closed",
}

def load_distraction_models(base_dir: Optional[str] = None):
    """Loads and returns YOLO, CNN, SVM, and Transforms."""
    if base_dir is None:
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(base_dir)

    model_dir = Path(os.environ.get("MODEL_DIR", str(base_dir / "models")))
    
    yolo_model, cnn, svm, scaler = None, None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLO
    try:
        from ultralytics import YOLO
        yolo_path = model_dir / "yolov8m.pt"
        if yolo_path.exists():
            yolo_model = YOLO(str(yolo_path))
    except Exception as e:
        print(f"Failed to load YOLO: {e}")

    # Load CNN
    try:
        pth_path = model_dir / "PretrainCNN_99.75.pth"
        cnn = models.resnet18()
        cnn.fc = nn.Identity()
        if pth_path.exists():
            state = torch.load(str(pth_path), map_location=device)
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            cnn.load_state_dict(state, strict=False)
            cnn.to(device).eval()
    except Exception as e:
        print(f"Failed to load CNN: {e}")
        cnn = None

    # Load SVM
    try:
        svm_path = model_dir / "PretrainCNNSVM.pkl"
        if svm_path.exists():
            svm_data = joblib.load(str(svm_path))
            if isinstance(svm_data, dict):
                svm, scaler = svm_data.get("svm"), svm_data.get("scaler")
            else:
                svm = svm_data
    except Exception as e:
        print(f"Failed to load SVM: {e}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return {"yolo": yolo_model, "cnn": cnn, "svm": svm, "scaler": scaler, "transform": transform, "device": device}


def _apply_face_heuristics(frame_bgr, label, confidence, face_cascade, eye_cascade):
    """Helper function to override false positives if the driver is looking forward."""
    if face_cascade is None or eye_cascade is None or "DISTRACTED" not in label:
        return label, confidence

    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Sort to get the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            
            # Check if face is centered in the frame
            img_w = frame_bgr.shape[1]
            face_center_x = fx + fw / 2
            
            if abs(face_center_x - img_w / 2) / img_w < 0.20:
                # Check for open eyes
                roi = gray[fy:fy + fh, fx:fx + fw]
                if len(eye_cascade.detectMultiScale(roi)) >= 1:
                    return "Normal Driving (OVERRIDDEN)", max(confidence, 0.95)
    except Exception:
        pass # If cascade fails, safely ignore and return original label
        
    return label, confidence


def predict_driver_behavior(frame_bgr, models, face_cascade=None, eye_cascade=None):
    global _frame_counter, _last_processed_frame, _last_metrics, _last_label
    _frame_counter += 1

    # ==========================================
    # PHASE 1: Frame Skipping (Use Cached Data)
    # ==========================================
    if _frame_counter % 3 != 0 and _last_processed_frame is not None:
        return _last_processed_frame, _last_metrics, _last_label

    if not models.get("yolo"):
        return frame_bgr, "YOLO model missing.", "No Model"

    start = time.perf_counter()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    final_frame = frame_bgr.copy() # We will draw boxes directly on this copy

    # ==========================================
    # PHASE 2: YOLO Detection & Cropping
    # ==========================================
    results = models["yolo"](rgb, classes=[0], verbose=False, device=models["device"])
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    label, confidence = "No Person Detected", 0.0

    if len(boxes) > 0:
        # Get the largest bounding box (the driver)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = map(int, boxes[idx])

        # Add 5% padding around the box
        pad_x, pad_y = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
        x1c, y1c = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2c, y2c = min(rgb.shape[1], x2 + pad_x), min(rgb.shape[0], y2 + pad_y)

        # ==========================================
        # PHASE 3: AI Classification (ResNet + SVM)
        # ==========================================
        person_crop = Image.fromarray(rgb[y1c:y2c, x1c:x2c])
        x = models["transform"](person_crop).unsqueeze(0).to(models["device"])

        with torch.no_grad():
            feat = models["cnn"](x).cpu().numpy()

        if models.get("svm"):
            if models.get("scaler"):
                feat = models["scaler"].transform(feat)
                
            probs = models["svm"].predict_proba(feat)[0]
            pred = int(np.argmax(probs))
            confidence = float(probs[pred])

            if confidence < 0.65 or (pred == 0 and confidence < 0.70):
                label = "Unknown Behavior"
            else:
                label = CLASS_LABELS.get(pred, "Unknown")
                if pred >= 1:
                    label += " (DISTRACTED)"

        # ==========================================
        # PHASE 4: Heuristic Overrides
        # ==========================================
        label, confidence = _apply_face_heuristics(final_frame, label, confidence, face_cascade, eye_cascade)

        # ==========================================
        # PHASE 5: Draw Results
        # ==========================================
        # Green (0,255,0) for normal, Red (0,0,255) for distracted
        box_color = (0, 255, 0) if ("Normal" in label or "Unknown" in label) else (0, 0, 255) 
        cv2.rectangle(final_frame, (x1, y1), (x2, y2), box_color, 4)

    # ==========================================
    # PHASE 6: Update Cache & Return
    # ==========================================
    latency = (time.perf_counter() - start) * 1000
    metrics_output = (f"Detected Behavior: {label}\n"
                      f"Confidence Score: {confidence*100:.2f}%\n"
                      f"Processing Latency: {latency:.2f}ms")

    _last_processed_frame = final_frame
    _last_metrics = metrics_output
    _last_label = label

    return final_frame, metrics_output, label