import cv2
import numpy as np
import onnxruntime as ort

# Paths
ONNX_MODEL_PATH = "E:/Face_Mask_Detection/jetson-inference/python/training/classification/models/face_mask/resnet18.onnx"
LABELS_PATH = "E:/Face_Mask_Detection/jetson-inference/python/training/classification/models/face_mask/labels.txt"

# Load ONNX model with explicit CPU provider
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
labels = [line.strip() for line in open(LABELS_PATH)]

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Preprocess function
def preprocess(face_image):
    img = cv2.resize(face_image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


# Predict function
def predict(face_image):
    input_tensor = preprocess(face_image)
    outputs = session.run(None, {"input_0": input_tensor})
    pred_idx = np.argmax(outputs[0])
    return labels[pred_idx]


# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        label = predict(face_img)

        # Set colors based on label
        if label.lower() == "mask" or label.lower() == "withmask":
            color = (0, 255, 0)  # Green for mask
        else:
            color = (0, 0, 255)  # Red for no mask

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()