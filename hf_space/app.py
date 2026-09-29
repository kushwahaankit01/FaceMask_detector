import os
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------------------------------------------
# ENVIRONMENT & PATHS
# ------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_DIR = os.path.join(BASE_DIR, "face_detector")
MASK_MODEL_H5_PATH = os.path.join(BASE_DIR, "mask_detector_new.h5")

prototxt_path = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
weights_path = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
print("[HF Space] Loading Face Detector...")
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

print("[HF Space] Loading Mask Detector (.h5)...")
mask_model_h5 = tf.keras.models.load_model(MASK_MODEL_H5_PATH, compile=False)

print("[HF Space] Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# INFERENCE LOGIC
# ------------------------------------------------------------
def filter_nested_boxes(locs):
    if len(locs) <= 1:
        return locs
    keep = []
    sorted_locs = sorted(locs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for box in sorted_locs:
        (sx1, sy1, ex1, ey1) = box
        area1 = (ex1 - sx1) * (ey1 - sy1)
        is_nested = False
        for (sx2, sy2, ex2, ey2) in keep:
            ix1, iy1 = max(sx1, sx2), max(sy1, sy2)
            ix2, iy2 = min(ex1, ex2), min(ey1, ey2)
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                if intersection / float(area1) > 0.7:
                    is_nested = True
                    break
        if not is_nested:
            keep.append(box)
    return keep

def process_frame(frame, conf_thresh=0.5, min_face=40):
    start_time = time.time()
    (h, w) = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    raw_locs = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < conf_thresh:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        if (endX - startX < min_face) or (endY - startY < min_face):
            continue
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)
        raw_locs.append((startX, startY, endX, endY))

    locs = filter_nested_boxes(raw_locs)

    faces = []
    valid_locs = []
    for (startX, startY, endX, endY) in locs:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        face_arr = img_to_array(face_rgb)
        face_arr = preprocess_input(face_arr)
        faces.append(face_arr)
        valid_locs.append((startX, startY, endX, endY))

    # MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    hand_boxes = []
    hand_landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs) * w), int(max(xs) * w)
            y_min, y_max = int(min(ys) * h), int(max(ys) * h)
            hand_boxes.append((x_min, y_min, x_max, y_max))
            pts = [{"x": round(lm.x * w, 1), "y": round(lm.y * h, 1)} for lm in hand_landmarks.landmark]
            hand_landmarks_list.append(pts)

    # Predictions
    results_list = []
    masked_count = 0
    no_mask_count = 0
    occluded_count = 0

    if len(faces) > 0:
        faces_np = np.array(faces, dtype="float32")
        preds = mask_model_h5.predict(faces_np, batch_size=16, verbose=0)

        for (box, pred) in zip(valid_locs, preds):
            (startX, startY, endX, endY) = box
            (mask_prob, without_mask_prob) = float(pred[0]), float(pred[1])

            occluded = any(
                hx1 < endX and hx2 > startX and hy1 < endY and hy2 > startY
                for (hx1, hy1, hx2, hy2) in hand_boxes
            )
            if occluded:
                occluded_count += 1

            if mask_prob > without_mask_prob:
                label = "Mask"
                confidence = mask_prob
                masked_count += 1
                color = [0, 230, 118]
            else:
                label = "No Mask"
                confidence = without_mask_prob
                no_mask_count += 1
                color = [255, 23, 68]

            results_list.append({
                "bbox": [int(startX), int(startY), int(endX), int(endY)],
                "label": label,
                "confidence": round(confidence * 100, 2),
                "hand_occluded": occluded,
                "color": color
            })

    latency_ms = round((time.time() - start_time) * 1000, 2)
    total_faces = len(results_list)
    compliance = round((masked_count / total_faces * 100), 1) if total_faces > 0 else 100.0

    return {
        "faces": results_list,
        "summary": {
            "total_faces": total_faces,
            "masked": masked_count,
            "no_mask": no_mask_count,
            "hand_occluded": occluded_count,
            "compliance_rate": compliance,
            "latency_ms": latency_ms
        },
        "hand_boxes": hand_boxes,
        "hand_landmarks": hand_landmarks_list
    }

def analyze_image_gradio(input_img):
    if input_img is None:
        return None, "No image provided"
    
    # Gradio provides RGB PIL or numpy
    if isinstance(input_img, np.ndarray):
        frame_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
        
    analysis = process_frame(frame_bgr)
    
    annotated = frame_bgr.copy()
    
    # Draw MediaPipe hand landmarks
    rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_results = hands_detector.process(rgb_img)
    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(254, 242, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(118, 230, 0), thickness=2)
            )

    for face in analysis["faces"]:
        (sx, sy, ex, ey) = face["bbox"]
        lbl = face["label"]
        conf = face["confidence"]
        color_bgr = (face["color"][2], face["color"][1], face["color"][0])
        if face["hand_occluded"]:
            lbl += " ✋ [Hand Occluded]"

        cv2.rectangle(annotated, (sx, sy), (ex, ey), color_bgr, 3)
        label_text = f"{lbl} ({conf}%)"
        cv2.rectangle(annotated, (sx, sy - 28), (sx + len(label_text) * 11, sy), color_bgr, -1)
        cv2.putText(annotated, label_text, (sx + 5, sy - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    for (hx1, hy1, hx2, hy2) in analysis["hand_boxes"]:
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (255, 234, 0), 2)
        cv2.putText(annotated, "HAND OCCLUSION DETECTED", (hx1, hy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 234, 0), 1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, analysis["summary"]

# ------------------------------------------------------------
# GRADIO INTERFACE (FREE HUGGING FACE SPACE SDK)
# ------------------------------------------------------------
demo = gr.Interface(
    fn=analyze_image_gradio,
    inputs=gr.Image(type="numpy", label="Upload Image or Webcam Capture"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Detection Output"),
        gr.JSON(label="Safety Compliance Analysis JSON")
    ],
    title="🏆 AegisVision AI — Safety Compliance Engine",
    description="3rd Place Winner - DSAI Hackathon. Real-Time Face Mask & MediaPipe Hand-Occlusion Engine.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
