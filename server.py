import os
import io
import time
import base64
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------------------------------------------
# ENVIRONMENT & PATHS SETUP
# ------------------------------------------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_DIR = os.path.join(BASE_DIR, "face_detector")
MASK_MODEL_H5_PATH = os.path.join(BASE_DIR, "mask_detector_new.h5")
MASK_MODEL_TFLITE_PATH = os.path.join(BASE_DIR, "mask_detector_quant.tflite")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

prototxt_path = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
weights_path = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------
print("[Server] Loading Face Detector (OpenCV SSD)...")
if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
    raise FileNotFoundError(f"Face detector files not found in {FACE_MODEL_DIR}")
faceNet = cv2.dnn.readNet(prototxt_path, weights_path)

print("[Server] Loading Mask Detector (.h5 Keras)...")
mask_model_h5 = None
if os.path.exists(MASK_MODEL_H5_PATH):
    mask_model_h5 = tf.keras.models.load_model(MASK_MODEL_H5_PATH, compile=False)

print("[Server] Loading TFLite Quantized Mask Detector...")
interpreter = None
input_details = None
output_details = None
if os.path.exists(MASK_MODEL_TFLITE_PATH):
    try:
        interpreter = tf.lite.Interpreter(model_path=MASK_MODEL_TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[Server] TFLite Quantized Model loaded successfully.")
    except Exception as e:
        print(f"[Warning] TFLite Interpreter load skipped: {e}. Fallback to Keras H5 model.")

print("[Server] Initializing MediaPipe Hands Detector...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------------------------
# FASTAPI APP INITIATION
# ------------------------------------------------------------
app = FastAPI(
    title="AegisVision AI - Safety Compliance Engine",
    description="Real-Time Face Mask & Hand-Occlusion Detection Backend API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def predict_mask_h5(faces_np):
    if mask_model_h5 is None:
        raise ValueError("H5 model not loaded")
    return mask_model_h5.predict(faces_np, batch_size=16, verbose=0)

def predict_mask_tflite(faces_np):
    if interpreter is None:
        return predict_mask_h5(faces_np)
    preds = []
    for face in faces_np:
        input_data = np.expand_dims(face, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output_data[0])
    return np.array(preds)

def filter_nested_boxes(locs):
    """Filter out smaller face boxes that are inside larger face boxes (e.g. open mouth)"""
    if len(locs) <= 1:
        return locs
    
    keep = []
    # Sort by area descending
    sorted_locs = sorted(locs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    
    for box in sorted_locs:
        (sx1, sy1, ex1, ey1) = box
        area1 = (ex1 - sx1) * (ey1 - sy1)
        is_nested = False
        
        for (sx2, sy2, ex2, ey2) in keep:
            # Calculate intersection
            ix1, iy1 = max(sx1, sx2), max(sy1, sy2)
            ix2, iy2 = min(ex1, ex2), min(ey1, ey2)
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # If more than 70% of box1 is inside box2, drop box1
                if intersection / float(area1) > 0.7:
                    is_nested = True
                    break
        if not is_nested:
            keep.append(box)
    return keep

def process_frame(frame, model_type="h5", conf_thresh=0.5, min_face=40):
    start_time = time.time()
    (h, w) = frame.shape[:2]

    # ---- 1. OpenCV SSD Face Detection ----
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

    # Filter nested face boxes (e.g. mouth detected as second face)
    locs = filter_nested_boxes(raw_locs)

    faces = []
    valid_locs = []

    for (startX, startY, endX, endY) in locs:
        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue

        # Convert OpenCV BGR -> RGB for MobileNetV2 preprocess_input
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        face_arr = img_to_array(face_rgb)
        face_arr = preprocess_input(face_arr)

        faces.append(face_arr)
        valid_locs.append((startX, startY, endX, endY))

    # ---- 2. MediaPipe Hand Detection (Occlusion Check & Landmarks) ----
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

    # ---- 3. Mask Model Prediction ----
    results_list = []
    masked_count = 0
    no_mask_count = 0
    uncertain_count = 0
    occluded_count = 0

    if len(faces) > 0:
        faces_np = np.array(faces, dtype="float32")
        
        # Use TFLite if available and requested, otherwise H5
        is_tflite = (model_type == "tflite" and interpreter is not None)
        if is_tflite:
            preds = predict_mask_tflite(faces_np)
            actual_model_name = "TFLITE"
        else:
            preds = predict_mask_h5(faces_np)
            actual_model_name = "H5"

        for (box, pred) in zip(valid_locs, preds):
            (startX, startY, endX, endY) = box
            
            # Correct Mapping in RGB:
            # Index 0: mask (With Mask)
            # Index 1: withoutMask (No Mask)
            (mask_prob, without_mask_prob) = float(pred[0]), float(pred[1])

            # Hand occlusion check
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
                color = [0, 230, 118] # Emerald Green
            else:
                label = "No Mask"
                confidence = without_mask_prob
                no_mask_count += 1
                color = [255, 23, 68] # Crimson Red

            if confidence < 0.5:
                label = "Uncertain"
                uncertain_count += 1
                color = [255, 196, 0] # Amber Yellow

            results_list.append({
                "bbox": [int(startX), int(startY), int(endX), int(endY)],
                "label": label,
                "confidence": round(confidence * 100, 2),
                "mask_probability": round(mask_prob * 100, 2),
                "no_mask_probability": round(without_mask_prob * 100, 2),
                "hand_occluded": occluded,
                "color": color
            })
    else:
        actual_model_name = model_type.upper()

    latency_ms = round((time.time() - start_time) * 1000, 2)
    total_faces = len(results_list)
    compliance_rate = round((masked_count / total_faces * 100), 1) if total_faces > 0 else 100.0

    return {
        "faces": results_list,
        "summary": {
            "total_faces": total_faces,
            "masked": masked_count,
            "no_mask": no_mask_count,
            "uncertain": uncertain_count,
            "hand_occluded": occluded_count,
            "compliance_rate": compliance_rate,
            "latency_ms": latency_ms,
            "model_used": actual_model_name
        },
        "hand_boxes": hand_boxes,
        "hand_landmarks": hand_landmarks_list
    }

# ------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>AegisVision API Server is Running! Frontend UI loading...</h2>")

@app.get("/api/status")
async def get_status():
    h5_size_mb = round(os.path.getsize(MASK_MODEL_H5_PATH) / (1024 * 1024), 2) if os.path.exists(MASK_MODEL_H5_PATH) else 0
    tflite_size_mb = round(os.path.getsize(MASK_MODEL_TFLITE_PATH) / (1024 * 1024), 2) if os.path.exists(MASK_MODEL_TFLITE_PATH) else 0
    return {
        "status": "online",
        "system": "AegisVision AI Engine",
        "models": {
            "h5_model": {
                "available": mask_model_h5 is not None,
                "name": "MobileNetV2 Keras (.h5)",
                "size_mb": h5_size_mb
            },
            "tflite_model": {
                "available": interpreter is not None,
                "name": "Quantized INT8 TFLite (.tflite)",
                "size_mb": tflite_size_mb,
                "size_reduction": f"{round((1 - tflite_size_mb/h5_size_mb)*100, 1)}%" if h5_size_mb else "N/A"
            },
            "face_detector": "OpenCV ResNet-SSD Caffe",
            "hand_tracker": "MediaPipe Hands v0.10"
        },
        # "hackathon_award": "🏆 3rd Place Winner - DSAI Hackathon"
    }

@app.post("/api/predict/frame")
async def predict_frame(
    file: UploadFile = File(...),
    model: str = Form("h5"),
    conf: float = Form(0.5)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        analysis = process_frame(img, model_type=model, conf_thresh=conf)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model: str = Form("h5"),
    draw_boxes: bool = Form(True)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        analysis = process_frame(img, model_type=model, conf_thresh=0.5)

        if draw_boxes:
            annotated = img.copy()

            # 1. Draw MediaPipe Hand Landmarks & Connections
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(rgb_img)
            if hand_results and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(254, 242, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(118, 230, 0), thickness=2)
                    )

            # 2. Draw Face Bounding Boxes
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

            # 3. Draw Hand Bounding Boxes if present
            for (hx1, hy1, hx2, hy2) in analysis["hand_boxes"]:
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (255, 234, 0), 2)
                cv2.putText(annotated, "HAND OCCLUSION DETECTED", (hx1, hy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 234, 0), 1)

            _, encoded_img = cv2.imencode(".jpg", annotated)
            base64_img = base64.b64encode(encoded_img).decode("utf-8")
            analysis["annotated_image"] = f"data:image/jpeg;base64,{base64_img}"

        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
