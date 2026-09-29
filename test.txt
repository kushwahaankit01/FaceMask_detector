# ============================================================
# REAL-TIME FACE MASK DETECTION FROM WEBCAM (VS Code + TFLite)
# ============================================================

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import imutils
import os

# ---- Paths ----
face_model_path = "D:/work/vidut_vega/mask_model/face_detector"
mask_model_path = "D:/work/vidut_vega/mask_model/mask_detector_quant.tflite"

# ---- Check if face model files exist ----
prototxtPath = os.path.join(face_model_path, "deploy.prototxt")
weightsPath = os.path.join(face_model_path, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxtPath):
    raise FileNotFoundError(f"Face prototxt not found: {prototxtPath}")
if not os.path.exists(weightsPath):
    raise FileNotFoundError(f"Face caffemodel not found: {weightsPath}")

# ---- Load Face Detector ----
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# ---- Load TFLite Mask Detector ----
if not os.path.exists(mask_model_path):
    raise FileNotFoundError(f"Mask TFLite model not found: {mask_model_path}")

interpreter = tf.lite.Interpreter(model_path=mask_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(interpreter, faces):
    preds = []
    for face in faces:
        input_data = np.expand_dims(face, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0])
    return np.array(preds)

# ---- Detection function ----
def detect_and_predict_mask(frame, faceNet, interpreter, confidence=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = predict_tflite(interpreter, faces)

    return locs, preds

# ---- Open Webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Cannot open webcam.")

print("🎥 Starting webcam feed. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    locs, preds = detect_and_predict_mask(frame, faceNet, interpreter)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        cv2.putText(frame, label_text, (startX, startY-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam feed stopped.")
