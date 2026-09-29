#!/usr/bin/env python3
"""
============================================================
SNAPSHOT-BASED FACE MASK + HAND DETECTION
Raspberry Pi 3 - Hackathon Edition
============================================================

Changes from original webcam version:
- Webcam → Pi Camera Module
- Real-time video → Snapshot capture (3-4 images)
- cv2.imshow() → Save to output folder
- Continuous loop → Triggered by face detection
- All features preserved: Mask + Hand detection + Occlusion
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
from datetime import datetime
import mediapipe as mp
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# ============================================================
# CONFIGURATION
# ============================================================

# Paths (adjusted for Raspberry Pi)
FACE_PROTOTXT = "models/deploy.prototxt"
FACE_WEIGHTS = "models/res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL = "models/mask_detector_quant.tflite"

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FRAMERATE = 20

# Detection parameters
NUM_SNAPSHOTS = 4              # Number of images to capture
FACE_CONFIDENCE = 0.5          # Face detection threshold
MASK_CONFIDENCE = 0.7          # Mask classification threshold
MIN_FACE_SIZE = 40             # Minimum face size in pixels
PADDING = 20                   # Padding around face box

# Output
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# BANNER
# ============================================================

print("=" * 60)
print(" SNAPSHOT-BASED MASK + HAND DETECTION")
print(" Raspberry Pi 3 - Hackathon Edition")
print("=" * 60)
print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

# ============================================================
# LOAD MODELS
# ============================================================

print("[1/3] Loading face detector...")
if not os.path.exists(FACE_PROTOTXT) or not os.path.exists(FACE_WEIGHTS):
    raise FileNotFoundError("Face detector model files missing!")
faceNet = cv2.dnn.readNet(FACE_PROTOTXT, FACE_WEIGHTS)
print("      ✓ Face detector loaded")

print()
print("[2/3] Loading TFLite mask detector...")
if not os.path.exists(MASK_MODEL):
    raise FileNotFoundError("Mask TFLite model not found!")
interpreter = tflite.Interpreter(model_path=MASK_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("      ✓ Mask detector loaded")

print()
print("[3/3] Loading MediaPipe hands detector...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=True,      # Changed to True for image processing
    max_num_hands=2,
    min_detection_confidence=0.5
)
print("      ✓ Hand detector loaded")

print()
print("=" * 60)
print(" ✓ ALL MODELS LOADED - READY TO CAPTURE")
print("=" * 60)
print()

# ============================================================
# INITIALIZE CAMERA
# ============================================================

print("Initializing Pi Camera...")
camera = PiCamera()
camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
camera.framerate = CAMERA_FRAMERATE
raw_capture = PiRGBArray(camera, size=(CAMERA_WIDTH, CAMERA_HEIGHT))
time.sleep(2)  # Camera warm-up
print("✓ Camera ready")
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_tflite(interpreter, faces):
    """Predict mask/no-mask using TFLite"""
    preds = []
    for face in faces:
        input_data = np.expand_dims(face, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0])
    return np.array(preds)


def detect_and_predict_mask(frame, faceNet, interpreter):
    """Detect faces and predict mask/no-mask"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs = [], []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < FACE_CONFIDENCE:
            continue
            
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Skip tiny faces
        if (endX - startX < MIN_FACE_SIZE) or (endY - startY < MIN_FACE_SIZE):
            continue

        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w - 1, endX), min(h - 1, endY)

        face = frame[startY:endY, startX:endX]
        if face.size == 0:
            continue
            
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((startX, startY, endX, endY, conf))

    preds = []
    if faces:
        faces = np.array(faces, dtype="float32")
        preds = predict_tflite(interpreter, faces)

    return locs, preds


def detect_hands(frame):
    """Detect hands using MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    hand_boxes = []
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            hand_boxes.append((x_min, y_min, x_max, y_max))

    return hand_boxes, results


def check_hand_occlusion(face_box, hand_boxes):
    """Check if hand is occluding face"""
    (fx1, fy1, fx2, fy2) = face_box[:4]
    
    for (hx1, hy1, hx2, hy2) in hand_boxes:
        if (hx1 < fx2 and hx2 > fx1 and hy1 < fy2 and hy2 > fy1):
            return True
    return False


def annotate_image(frame, face_locations, mask_predictions, hand_boxes, hand_landmarks):
    """Draw all detections on image"""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw hand landmarks
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            )

    # Draw hand bounding boxes
    for (hx1, hy1, hx2, hy2) in hand_boxes:
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
        cv2.putText(annotated, "HAND", (hx1, hy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Process faces (sorted left to right)
    face_data = list(zip(face_locations, mask_predictions))
    face_data_sorted = sorted(face_data, key=lambda x: x[0][0])  # Sort by startX

    for person_num, (location, prediction) in enumerate(face_data_sorted, start=1):
        (startX, startY, endX, endY, face_conf) = location
        (mask, withoutMask) = prediction

        # Apply padding
        startX = max(0, startX - PADDING)
        startY = max(0, startY - PADDING)
        endX = min(w - 1, endX + PADDING)
        endY = min(h - 1, endY + PADDING)

        # Check occlusion
        occluded = check_hand_occlusion(location, hand_boxes)

        # Determine label
        if mask > withoutMask and mask > MASK_CONFIDENCE:
            label = "Mask"
            color = (0, 255, 0)
        elif withoutMask > MASK_CONFIDENCE:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Uncertain"
            color = (0, 255, 255)

        if occluded:
            label += " (Hand Occluded)"
            color = (0, 165, 255)  # Orange

        confidence = max(mask, withoutMask) * 100

        # Draw bounding box
        cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)

        # Draw label with background
        label_text = f"{label}: {confidence:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            annotated,
            (startX, startY - text_h - 10),
            (startX + text_w, startY),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (startX, startY - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Person number
        cv2.putText(
            annotated,
            f"Person #{person_num}",
            (startX, endY + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def save_snapshot(image, snapshot_num, detection_info):
    """Save annotated image and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}_{snapshot_num}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(filepath, image)

    # Save metadata
    metadata_file = filepath.replace(".jpg", "_info.txt")
    with open(metadata_file, "w") as f:
        f.write(f"Snapshot #{snapshot_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}\n")
        f.write(f"\nDetections:\n")
        for info in detection_info:
            f.write(f"  - {info}\n")

    return filename

# ============================================================
# MAIN DETECTION LOOP
# ============================================================

print("=" * 60)
print(" STARTING SNAPSHOT CAPTURE")
print(f" Will capture {NUM_SNAPSHOTS} images when faces detected")
print(" Press Ctrl+C to stop")
print("=" * 60)
print()

snapshot_count = 0
face_wait_frames = 0

try:
    for frame_data in camera.capture_continuous(
        raw_capture, format="bgr", use_video_port=True
    ):
        frame = frame_data.array

        # Quick face check
        locs, _ = detect_and_predict_mask(frame, faceNet, interpreter)

        if len(locs) > 0:
            face_wait_frames += 1

            # Wait for stable face detection
            if face_wait_frames >= 5:
                snapshot_count += 1
                print(f"\n[Snapshot {snapshot_count}/{NUM_SNAPSHOTS}] Face detected! Processing...")

                # Full pipeline
                print("  → Detecting faces...")
                face_locs, mask_preds = detect_and_predict_mask(frame, faceNet, interpreter)
                print(f"     Found {len(face_locs)} face(s)")

                print("  → Classifying masks...")
                
                print("  → Detecting hands...")
                hand_boxes, hand_results = detect_hands(frame)
                hand_landmarks = (
                    hand_results.multi_hand_landmarks
                    if hand_results.multi_hand_landmarks
                    else []
                )
                print(f"     Found {len(hand_boxes)} hand(s)")

                # Collect detection info (sorted left to right)
                detection_info = []
                face_data = list(zip(face_locs, mask_preds))
                face_data_sorted = sorted(face_data, key=lambda x: x[0][0])

                for person_num, (loc, pred) in enumerate(face_data_sorted, start=1):
                    (mask, withoutMask) = pred
                    label = "MASK" if mask > withoutMask else "NO MASK"
                    conf = max(mask, withoutMask) * 100
                    occluded = check_hand_occlusion(loc, hand_boxes)

                    info = f"Person {person_num} (from left): {label} ({conf:.1f}%)"
                    if occluded:
                        info += " - Hand occluding face"
                    detection_info.append(info)
                    print(f"     {info}")

                # Annotate image
                print("  → Annotating image...")
                annotated_frame = annotate_image(
                    frame, face_locs, mask_preds, hand_boxes, hand_landmarks
                )

                # Save
                print("  → Saving to output folder...")
                filename = save_snapshot(annotated_frame, snapshot_count, detection_info)
                print(f"     ✓ Saved: {filename}")

                # Wait before next snapshot
                time.sleep(2)
                face_wait_frames = 0

                # Check if done
                if snapshot_count >= NUM_SNAPSHOTS:
                    print()
                    print("=" * 60)
                    print(f" ✓ CAPTURED {NUM_SNAPSHOTS} SNAPSHOTS")
                    print("=" * 60)
                    break
        else:
            face_wait_frames = 0

        raw_capture.truncate(0)

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    camera.close()
    hands_detector.close()

    print()
    print("=" * 60)
    print(" DETECTION COMPLETE")
    print("=" * 60)
    print(f" Total snapshots captured: {snapshot_count}")
    print(f" Output location: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)
    print()
